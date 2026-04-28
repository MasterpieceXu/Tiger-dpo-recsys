"""
Evaluation module: TIGER (SFT and SFT+DPO) vs. classical baselines.

Key differences from the original implementation
------------------------------------------------
1. ``BaselineRecommender`` no longer materializes a dense ``items x items``
   cosine-similarity matrix. On ml-32m (~70k items) that was a 39 GB float64
   matrix that OOMed every machine. We use a sparse user-item matrix and
   ``sklearn.neighbors.NearestNeighbors`` with ``knn_top_n`` neighbors per
   item instead. Memory budget drops from O(I^2) to O(I * knn_top_n).

2. ``RecommendationEvaluator`` accepts ``max_test_users`` so we can subsample
   the test set during evaluation (the cosine-based baselines are still
   linear in the number of test users).

3. ``run_evaluation`` returns a structured dict that the new ``src/report.py``
   converts into a Markdown table for the README.

4. ``evaluate_tiger_models`` evaluates both the SFT-only and the SFT+DPO
   policies if both checkpoints are present, so the final report can show
   the DPO ablation row.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# project root on path so imports work in any launch mode
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.tiger_model import TIGERModel  # noqa: E402
from config import Config  # noqa: E402
from utils import calculate_metrics, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class BaselineRecommender:
    """Memory-efficient classical baselines: Popular / ItemKNN / Random.

    The user-item matrix is built as a CSR sparse matrix; ItemKNN uses
    ``sklearn.neighbors.NearestNeighbors(metric='cosine')`` with
    ``n_neighbors = knn_top_n + 1`` (the +1 is the item itself).
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        knn_top_n: int = 50,
    ) -> None:
        self.ratings_df = ratings_df
        self.knn_top_n = knn_top_n

        self.users: np.ndarray = ratings_df["userId"].unique()
        self.items: np.ndarray = ratings_df["movieId"].unique()

        # Index lookups for sparse matrix construction.
        self._user_to_idx = {u: i for i, u in enumerate(self.users)}
        self._item_to_idx = {it: i for i, it in enumerate(self.items)}
        self._idx_to_item = {i: it for it, i in self._item_to_idx.items()}

        # Sparse user-item interaction matrix. We use binary positive signal
        # (rating >= min_rating already filtered upstream) which is enough for
        # popularity / cosine similarity baselines.
        rows = ratings_df["userId"].map(self._user_to_idx).values
        cols = ratings_df["movieId"].map(self._item_to_idx).values
        data = np.ones(len(ratings_df), dtype=np.float32)
        self.user_item: csr_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.users), len(self.items)),
            dtype=np.float32,
        )

        # Per-item popularity (nnz column counts).
        self._popularity = np.asarray(self.user_item.sum(axis=0)).ravel()
        self._popular_order = np.argsort(-self._popularity)  # most -> least

        # ItemKNN model: fit lazily only if used (it's the most expensive part).
        self._knn: Optional[NearestNeighbors] = None
        self._neighbor_idx: Optional[np.ndarray] = None      # [I, knn_top_n]
        self._neighbor_sim: Optional[np.ndarray] = None      # [I, knn_top_n]

        logger.info(
            "BaselineRecommender ready: %d users x %d items (sparse, %d nnz)",
            self.user_item.shape[0],
            self.user_item.shape[1],
            self.user_item.nnz,
        )

    # ----- popularity ----------------------------------------------------

    def _user_seen_items(self, user_id: int) -> set:
        if user_id not in self._user_to_idx:
            return set()
        u = self._user_to_idx[user_id]
        seen_idx = self.user_item[u].indices
        return {self._idx_to_item[i] for i in seen_idx}

    def recommend_popular(self, user_id: int, k: int = 50, exclude_seen: bool = True) -> List[int]:
        seen = self._user_seen_items(user_id) if exclude_seen else set()
        out: List[int] = []
        for idx in self._popular_order:
            item = self._idx_to_item[idx]
            if item in seen:
                continue
            out.append(item)
            if len(out) >= k:
                break
        return out

    def recommend_random(self, user_id: int, k: int = 50, exclude_seen: bool = True) -> List[int]:
        seen = self._user_seen_items(user_id) if exclude_seen else set()
        candidates = [self._idx_to_item[i] for i in range(len(self.items))
                      if self._idx_to_item[i] not in seen]
        return random.sample(candidates, min(k, len(candidates)))

    # ----- ItemKNN -------------------------------------------------------

    def _fit_itemknn(self) -> None:
        if self._knn is not None:
            return
        logger.info("Fitting ItemKNN (NearestNeighbors, top-%d)...", self.knn_top_n)
        # We want neighbors for each *item*, so transpose to items x users.
        item_user = self.user_item.T.tocsr()
        n_neighbors = min(self.knn_top_n + 1, item_user.shape[0])
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
        knn.fit(item_user)

        # Pre-compute the neighbor matrix: shape [I, knn_top_n]. The first
        # column is the item itself, drop it.
        distances, indices = knn.kneighbors(item_user)
        sim = 1.0 - distances
        self._neighbor_idx = indices[:, 1:]
        self._neighbor_sim = sim[:, 1:]
        self._knn = knn
        logger.info(
            "ItemKNN ready: neighbor matrix shape %s (~%.1f MB)",
            self._neighbor_idx.shape,
            self._neighbor_idx.nbytes / 1024**2,
        )

    def recommend_itemknn(
        self,
        user_id: int,
        k: int = 50,
        exclude_seen: bool = True,
    ) -> List[int]:
        self._fit_itemknn()
        if user_id not in self._user_to_idx:
            return self.recommend_popular(user_id, k, exclude_seen)

        u = self._user_to_idx[user_id]
        seen_idx = self.user_item[u].indices
        if len(seen_idx) == 0:
            return self.recommend_popular(user_id, k, exclude_seen)

        # Aggregate neighbor scores for items the user has interacted with.
        scores = np.zeros(len(self.items), dtype=np.float32)
        nbr_idx = self._neighbor_idx[seen_idx]                   # [|seen|, K]
        nbr_sim = self._neighbor_sim[seen_idx]                   # [|seen|, K]
        np.add.at(scores, nbr_idx.ravel(), nbr_sim.ravel())

        if exclude_seen:
            scores[seen_idx] = -np.inf

        top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [self._idx_to_item[int(i)] for i in top_idx]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RecommendationEvaluator:
    """Evaluate TIGER (any number of variants) and the classical baselines."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        random.seed(config.seed)
        np.random.seed(config.seed)

    # ----- I/O helpers --------------------------------------------------

    def _load_test_sequences(self, sequences_dir: str) -> Dict[int, List[str]]:
        path = os.path.join(sequences_dir, "test_sequences.json")
        with open(path, "r") as f:
            sequences = json.load(f)
        sequences = {int(k): v for k, v in sequences.items()}
        if self.config.eval.max_test_users and len(sequences) > self.config.eval.max_test_users:
            keys = random.sample(list(sequences.keys()), self.config.eval.max_test_users)
            sequences = {k: sequences[k] for k in keys}
            logger.info("Sampled %d test users (cap = %d)", len(sequences),
                        self.config.eval.max_test_users)
        else:
            logger.info("Using %d test users (no cap reached)", len(sequences))
        return sequences

    def _load_semantic_id_mapping(self, data_dir: str) -> Dict[Tuple[int, ...], int]:
        path = os.path.join(data_dir, "item_semantic_ids.jsonl")
        mapping: Dict[Tuple[int, ...], int] = {}
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                mapping[tuple(item["semantic_ids"])] = item["movieId"]
        return mapping

    # ----- target / prediction extraction -------------------------------

    @staticmethod
    def _tokens_to_movie_id(
        tokens: List[str],
        semantic_to_movie: Dict[Tuple[int, ...], int],
    ) -> Optional[int]:
        ids: List[int] = []
        for tok in tokens:
            if tok.startswith("<id_") and tok.endswith(">"):
                try:
                    ids.append(int(tok[4:-1]))
                except ValueError:
                    continue
        return semantic_to_movie.get(tuple(ids))

    # ----- TIGER --------------------------------------------------------

    def evaluate_tiger_model(
        self,
        model_path: str,
        test_sequences: Dict[int, List[str]],
        semantic_to_movie: Dict[Tuple[int, ...], int],
        k_values: List[int],
    ) -> Dict[str, float]:
        logger.info("Evaluating TIGER checkpoint at %s", model_path)
        model = TIGERModel.from_pretrained(model_path).to(self.device).eval()

        predictions: List[List[int]] = []
        ground_truth: List[List[int]] = []

        for user_id, sequence in test_sequences.items():
            if len(sequence) < 4:
                continue
            input_seq = sequence[:-2]
            last_two = sequence[-2:]

            try:
                semantic_recs = model.recommend(
                    input_seq,
                    num_recommendations=max(k_values),
                    num_beams=20,
                )
            except Exception as exc:
                logger.warning("recommend() failed for user %s: %s", user_id, exc)
                predictions.append([])
                ground_truth.append([])
                continue

            movie_recs: List[int] = []
            for rec in semantic_recs:
                key = tuple(rec)
                if key in semantic_to_movie:
                    mid = semantic_to_movie[key]
                    if mid not in movie_recs:
                        movie_recs.append(mid)
            predictions.append(movie_recs[: max(k_values)])

            target_movie = self._tokens_to_movie_id(last_two, semantic_to_movie)
            ground_truth.append([target_movie] if target_movie is not None else [])

        return calculate_metrics(predictions, ground_truth, k_values)

    def evaluate_tiger_variants(
        self,
        model_paths: Dict[str, str],
        test_sequences: Dict[int, List[str]],
        semantic_to_movie: Dict[Tuple[int, ...], int],
        k_values: List[int],
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for label, path in model_paths.items():
            if not os.path.isdir(path):
                logger.warning("Skip %s: path %s not found", label, path)
                continue
            results[label] = self.evaluate_tiger_model(
                path, test_sequences, semantic_to_movie, k_values
            )
        return results

    # ----- baselines ----------------------------------------------------

    def evaluate_baselines(
        self,
        ratings_df: pd.DataFrame,
        test_sequences: Dict[int, List[str]],
        semantic_to_movie: Dict[Tuple[int, ...], int],
        k_values: List[int],
    ) -> Dict[str, Dict[str, float]]:
        baseline = BaselineRecommender(ratings_df, knn_top_n=self.config.eval.knn_top_n)

        ground_truth: List[List[int]] = []
        users_evaluated: List[int] = []
        for user_id, sequence in test_sequences.items():
            if len(sequence) < 2:
                continue
            users_evaluated.append(user_id)
            target = self._tokens_to_movie_id(sequence[-2:], semantic_to_movie)
            ground_truth.append([target] if target is not None else [])

        methods = {
            "Popular": baseline.recommend_popular,
            "ItemKNN": baseline.recommend_itemknn,
            "Random": baseline.recommend_random,
        }
        out: Dict[str, Dict[str, float]] = {}
        for name, fn in methods.items():
            logger.info("Evaluating baseline: %s", name)
            preds: List[List[int]] = []
            for uid in users_evaluated:
                try:
                    preds.append(fn(uid, k=max(k_values)))
                except Exception as exc:
                    logger.warning("%s failed for user %s: %s", name, uid, exc)
                    preds.append([])
            out[name] = calculate_metrics(preds, ground_truth, k_values)
        return out

    # ----- driver -------------------------------------------------------

    def run_evaluation(
        self,
        sequences_dir: str,
        data_dir: str,
        tiger_model_paths: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run the full evaluation and persist ``evaluation_results.json``.

        Parameters
        ----------
        tiger_model_paths
            Mapping from a display label (e.g. ``"TIGER (SFT)"``,
            ``"TIGER + DPO"``) to a checkpoint directory. Missing paths are
            skipped silently.
        """
        if tiger_model_paths is None:
            tiger_model_paths = {
                "TIGER (SFT)": os.path.join(self.config.model_dir, "tiger_final"),
                "TIGER + DPO": os.path.join(self.config.model_dir, "onerec_lite_dpo"),
            }

        test_sequences = self._load_test_sequences(sequences_dir)
        semantic_to_movie = self._load_semantic_id_mapping(data_dir)
        ratings_df = pd.read_csv(os.path.join(data_dir, "processed_ratings.csv"))
        k_values = self.config.eval.recall_k

        tiger_results = self.evaluate_tiger_variants(
            tiger_model_paths, test_sequences, semantic_to_movie, k_values
        )
        baseline_results = self.evaluate_baselines(
            ratings_df, test_sequences, semantic_to_movie, k_values
        )

        results: Dict[str, Dict[str, float]] = {**tiger_results, **baseline_results}

        out_path = os.path.join(self.config.output_dir, "evaluation_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results written to %s", out_path)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config()
    setup_logging(os.path.join(config.log_dir, "evaluation.log"))

    evaluator = RecommendationEvaluator(config)
    sequences_dir = os.path.join(config.output_dir, "sequences")
    results = evaluator.run_evaluation(sequences_dir, config.output_dir)

    print("\nEvaluation Results")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

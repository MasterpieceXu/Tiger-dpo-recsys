"""
Evaluation module for recommendation models
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import random

# Add parent directory to path so `src.*`, `config`, and `utils` resolve no
# matter how this module is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.tiger_model import TIGERModel  # noqa: E402
from config import Config  # noqa: E402
from utils import calculate_metrics, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

class BaselineRecommender:
    """Baseline recommendation models for comparison"""
    
    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.user_item_matrix = None
        self.item_similarity = None
        self.popularity_scores = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for baseline models"""
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        # Calculate item popularity
        self.popularity_scores = self.ratings_df['movieId'].value_counts().to_dict()
        
        # Calculate item similarity (for ItemKNN)
        item_features = self.user_item_matrix.T.values
        self.item_similarity = cosine_similarity(item_features)
        
        logger.info("Baseline data preparation completed")
    
    def recommend_popular(self, user_id: int, k: int = 50, 
                         exclude_seen: bool = True) -> List[int]:
        """Popularity-based recommendation"""
        if exclude_seen and user_id in self.user_item_matrix.index:
            seen_items = set(self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index)
        else:
            seen_items = set()
        
        # Get top popular items
        popular_items = [
            item for item, _ in sorted(
                self.popularity_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            if item not in seen_items
        ]
        
        return popular_items[:k]
    
    def recommend_itemknn(self, user_id: int, k: int = 50, 
                         n_neighbors: int = 20, exclude_seen: bool = True) -> List[int]:
        """Item-based collaborative filtering"""
        if user_id not in self.user_item_matrix.index:
            return self.recommend_popular(user_id, k, exclude_seen)
        
        user_ratings = self.user_item_matrix.loc[user_id]
        seen_items = set(user_ratings[user_ratings > 0].index)
        
        if exclude_seen:
            candidate_items = [
                item for item in self.user_item_matrix.columns 
                if item not in seen_items
            ]
        else:
            candidate_items = list(self.user_item_matrix.columns)
        
        # Calculate scores for candidate items
        item_scores = {}
        for candidate in candidate_items:
            candidate_idx = list(self.user_item_matrix.columns).index(candidate)
            
            # Find similar items that user has rated
            similarities = []
            for seen_item in seen_items:
                seen_idx = list(self.user_item_matrix.columns).index(seen_item)
                sim = self.item_similarity[candidate_idx][seen_idx]
                rating = user_ratings[seen_item]
                similarities.append(sim * rating)
            
            if similarities:
                item_scores[candidate] = np.mean(similarities)
            else:
                item_scores[candidate] = 0
        
        # Sort and return top k
        recommended_items = sorted(
            item_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [item for item, _ in recommended_items[:k]]
    
    def recommend_random(self, user_id: int, k: int = 50, 
                        exclude_seen: bool = True) -> List[int]:
        """Random recommendation"""
        if exclude_seen and user_id in self.user_item_matrix.index:
            seen_items = set(self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index)
        else:
            seen_items = set()
        
        candidate_items = [
            item for item in self.user_item_matrix.columns 
            if item not in seen_items
        ]
        
        return random.sample(candidate_items, min(k, len(candidate_items)))

class RecommendationEvaluator:
    """Evaluator for recommendation models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
    def load_test_data(self, sequences_dir: str) -> Dict[str, List[str]]:
        """Load test sequences"""
        test_path = os.path.join(sequences_dir, "test_sequences.json")
        with open(test_path, 'r') as f:
            test_sequences = json.load(f)
        
        # Convert string keys to int
        test_sequences = {int(k): v for k, v in test_sequences.items()}
        
        logger.info(f"Loaded test data for {len(test_sequences)} users")
        return test_sequences
    
    def load_semantic_id_mapping(self, data_dir: str) -> Dict[Tuple[int, ...], int]:
        """Load semantic ID sequence (tuple) to movie ID mapping"""
        semantic_ids_path = os.path.join(data_dir, "item_semantic_ids.jsonl")
        semantic_to_movie: Dict[Tuple[int, ...], int] = {}
        
        with open(semantic_ids_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                movie_id = item['movieId']
                # 使用完整的语义 ID 序列作为 key，例如 (a, b)
                semantic_ids = tuple(item['semantic_ids'])
                semantic_to_movie[semantic_ids] = movie_id
        
        return semantic_to_movie
    
    def convert_semantic_to_movie_ids(self, semantic_recommendations: List[List[int]], 
                                    semantic_to_movie: Dict[Tuple[int, ...], int]) -> List[List[int]]:
        """Convert semantic IDs to movie IDs"""
        movie_recommendations = []
        
        for semantic_ids in semantic_recommendations:
            movie_recs: List[int] = []
            key = tuple(semantic_ids)
            if key in semantic_to_movie:
                movie_id = semantic_to_movie[key]
                movie_recs.append(movie_id)
            movie_recommendations.append(movie_recs)
        
        return movie_recommendations
    
    def evaluate_tiger_model(self, model_path: str, test_sequences: Dict[int, List[str]], 
                           semantic_to_movie: Dict[Tuple[int, ...], int], k_values: List[int]) -> Dict[str, float]:
        """Evaluate TIGER model"""
        logger.info("Evaluating TIGER model...")
        
        # Load model
        model = TIGERModel.from_pretrained(model_path)
        model.eval()
        model.to(self.device)
        
        predictions = []
        ground_truth = []
        
        for user_id, sequence in test_sequences.items():
            if len(sequence) < 2:
                continue
            
            # 使用除最后一部电影外的历史作为输入：这里假设每部电影由两个语义 token 表示
            if len(sequence) < 4:
                # 序列太短，无法构成至少 1 部电影历史 + 1 部电影目标
                continue
            input_sequence = sequence[:-2]
            last_tokens = sequence[-2:]
            
            # Generate recommendations
            try:
                semantic_recs = model.recommend(
                    input_sequence, 
                    num_recommendations=max(k_values),
                    num_beams=20
                )
                
                # Convert to movie IDs
                movie_recs = []
                for semantic_rec in semantic_recs:
                    key = tuple(semantic_rec)
                    if key in semantic_to_movie:
                        movie_id = semantic_to_movie[key]
                        if movie_id not in movie_recs:
                            movie_recs.append(movie_id)
                
                predictions.append(movie_recs[:max(k_values)])
                
                # Convert ground truth
                true_movie_ids = []
                semantic_ids: List[int] = []
                for token in last_tokens:
                    if token.startswith('<id_') and token.endswith('>'):
                        try:
                            semantic_ids.append(int(token[4:-1]))
                        except ValueError:
                            continue
                if semantic_ids:
                    key = tuple(semantic_ids)
                    if key in semantic_to_movie:
                        true_movie_ids.append(semantic_to_movie[key])
                
                ground_truth.append(true_movie_ids)
                
            except Exception as e:
                logger.warning(f"Error generating recommendations for user {user_id}: {e}")
                predictions.append([])
                ground_truth.append([])
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truth, k_values)
        
        logger.info("TIGER evaluation completed")
        return metrics
    
    def evaluate_baselines(self, ratings_df: pd.DataFrame, test_sequences: Dict[int, List[str]], 
                          semantic_to_movie: Dict[Tuple[int, ...], int], k_values: List[int]) -> Dict[str, Dict[str, float]]:
        """Evaluate baseline models"""
        logger.info("Evaluating baseline models...")
        
        baseline = BaselineRecommender(ratings_df)
        results = {}
        
        # Prepare test data
        test_users = []
        ground_truth = []
        
        for user_id, sequence in test_sequences.items():
            if len(sequence) < 2:
                continue
            
            test_users.append(user_id)
            
            # Convert ground truth
            true_movie_ids = []
            if len(sequence) >= 2:
                last_tokens = sequence[-2:]
                semantic_ids: List[int] = []
                for token in last_tokens:
                    if token.startswith('<id_') and token.endswith('>'):
                        try:
                            semantic_ids.append(int(token[4:-1]))
                        except ValueError:
                            continue
                if semantic_ids:
                    key = tuple(semantic_ids)
                    if key in semantic_to_movie:
                        true_movie_ids.append(semantic_to_movie[key])
            
            ground_truth.append(true_movie_ids)
        
        # Evaluate each baseline
        baseline_methods = {
            'Popular': baseline.recommend_popular,
            'ItemKNN': baseline.recommend_itemknn,
            'Random': baseline.recommend_random
        }
        
        for method_name, method_func in baseline_methods.items():
            logger.info(f"Evaluating {method_name}...")
            
            predictions = []
            for user_id in test_users:
                try:
                    recs = method_func(user_id, k=max(k_values))
                    predictions.append(recs)
                except Exception as e:
                    logger.warning(f"Error in {method_name} for user {user_id}: {e}")
                    predictions.append([])
            
            # Calculate metrics
            metrics = calculate_metrics(predictions, ground_truth, k_values)
            results[method_name] = metrics
        
        logger.info("Baseline evaluation completed")
        return results
    
    def run_evaluation(self, model_path: str, sequences_dir: str, data_dir: str) -> Dict:
        """Run complete evaluation"""
        logger.info("Starting complete evaluation...")
        
        # Load test data
        test_sequences = self.load_test_data(sequences_dir)
        
        # Load semantic ID mapping
        semantic_to_movie = self.load_semantic_id_mapping(data_dir)
        
        # Load ratings for baselines
        ratings_path = os.path.join(data_dir, "processed_ratings.csv")
        ratings_df = pd.read_csv(ratings_path)
        
        k_values = self.config.eval.recall_k
        
        # Evaluate TIGER model
        tiger_metrics = self.evaluate_tiger_model(
            model_path, test_sequences, semantic_to_movie, k_values
        )
        
        # Evaluate baselines
        baseline_metrics = self.evaluate_baselines(
            ratings_df, test_sequences, semantic_to_movie, k_values
        )
        
        # Combine results
        results = {
            'TIGER': tiger_metrics,
            **baseline_metrics
        }
        
        # Save results
        results_path = os.path.join(self.config.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        return results

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "evaluation.log"))
    
    evaluator = RecommendationEvaluator(config)
    
    # Run evaluation
    model_path = os.path.join(config.model_dir, "tiger_final")
    sequences_dir = os.path.join(config.output_dir, "sequences")
    
    results = evaluator.run_evaluation(model_path, sequences_dir, config.output_dir)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    print("\nEvaluation completed!")

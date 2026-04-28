"""
Data preprocessing for MovieLens-32M dataset
"""
import os
import sys
import re
import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Make `utils` (which lives at the project root) importable whether this file is
# run directly (`python src/data_preprocessing.py`) or imported as `src.data_preprocessing`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import load_movielens_data, filter_data  # noqa: E402

logger = logging.getLogger(__name__)

class MovieLensPreprocessor:
    """Preprocessor for MovieLens data"""
    
    def __init__(self, data_dir: str, min_rating: float = 4.0, min_interactions: int = 5):
        self.data_dir = data_dir
        self.min_rating = min_rating
        self.min_interactions = min_interactions
        self.tfidf_vectorizer = None
        self.movie_encoder = None
        self.user_encoder = None
        
    def load_and_filter_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and filter MovieLens data"""
        logger.info("Loading MovieLens data...")
        ratings, movies = load_movielens_data(self.data_dir)
        
        logger.info(f"Original data: {len(ratings)} ratings, {len(movies)} movies")
        
        # Filter ratings
        filtered_ratings = filter_data(ratings, self.min_rating, self.min_interactions)
        
        # Keep only movies that appear in filtered ratings
        valid_movies = movies[movies['movieId'].isin(filtered_ratings['movieId'].unique())]
        
        logger.info(f"Filtered data: {len(filtered_ratings)} ratings, {len(valid_movies)} movies")
        
        return filtered_ratings, valid_movies
    
    def create_item_corpus(self, movies: pd.DataFrame) -> List[str]:
        """Create text corpus for movies"""
        logger.info("Creating item corpus...")
        
        corpus = []
        for _, movie in movies.iterrows():
            # Extract title and year
            title = movie['title']
            year_match = re.search(r'\((\d{4})\)', title)
            year = year_match.group(1) if year_match else "unknown"
            clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip()
            
            # Extract genres
            genres = movie['genres'].replace('|', ' ')
            
            # Combine all text features
            text = f"{clean_title} {genres} {year}"
            corpus.append(text)
        
        return corpus
    
    def fit_tfidf(self, corpus: List[str], max_features: int = 10000) -> np.ndarray:
        """Fit TF-IDF vectorizer and transform corpus"""
        logger.info("Fitting TF-IDF vectorizer...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        return tfidf_matrix.toarray()
    
    def encode_movies_and_users(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        """Encode movie and user IDs"""
        logger.info("Encoding movie and user IDs...")
        
        # Encode movies
        self.movie_encoder = LabelEncoder()
        movies['encoded_movieId'] = self.movie_encoder.fit_transform(movies['movieId'])
        
        # Encode users
        self.user_encoder = LabelEncoder()
        ratings['encoded_userId'] = self.user_encoder.fit_transform(ratings['userId'])
        
        # Create mapping dictionaries
        movie_id_to_encoded = dict(zip(movies['movieId'], movies['encoded_movieId']))
        ratings['encoded_movieId'] = ratings['movieId'].map(movie_id_to_encoded)
        
        return ratings, movies
    
    def save_corpus(self, movies: pd.DataFrame, corpus: List[str], tfidf_features: np.ndarray, 
                   output_path: str):
        """Save corpus with features"""
        logger.info(f"Saving corpus to {output_path}...")
        
        corpus_data = []
        for i, (_, movie) in enumerate(movies.iterrows()):
            corpus_data.append({
                'movieId': int(movie['movieId']),
                'encoded_movieId': int(movie['encoded_movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'text': corpus[i],
                'tfidf_features': tfidf_features[i].tolist()
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in corpus_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def process_data(self, output_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Complete data processing pipeline"""
        # Load and filter data
        ratings, movies = self.load_and_filter_data()
        
        # Encode IDs
        ratings, movies = self.encode_movies_and_users(ratings, movies)
        
        # Create corpus
        corpus = self.create_item_corpus(movies)
        
        # Fit TF-IDF
        tfidf_features = self.fit_tfidf(corpus)
        
        # Save corpus
        import os
        os.makedirs(output_dir, exist_ok=True)
        corpus_path = os.path.join(output_dir, "item_corpus.jsonl")
        self.save_corpus(movies, corpus, tfidf_features, corpus_path)
        
        # Save processed data
        ratings.to_csv(os.path.join(output_dir, "processed_ratings.csv"), index=False)
        movies.to_csv(os.path.join(output_dir, "processed_movies.csv"), index=False)
        
        logger.info("Data processing completed!")
        
        return ratings, movies, tfidf_features

def load_corpus(corpus_path: str) -> Tuple[List[Dict], np.ndarray]:
    """Load corpus from file"""
    corpus_data = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus_data.append(json.loads(line))
    
    # Extract TF-IDF features
    tfidf_features = np.array([item['tfidf_features'] for item in corpus_data])
    
    return corpus_data, tfidf_features

if __name__ == "__main__":
    from config import Config
    from utils import setup_logging
    
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "preprocessing.log"))
    
    preprocessor = MovieLensPreprocessor(
        data_dir=config.data.data_dir,
        min_rating=config.data.min_rating,
        min_interactions=config.data.min_interactions
    )
    
    ratings, movies, tfidf_features = preprocessor.process_data(config.output_dir)
    
    print(f"Processed {len(ratings)} ratings and {len(movies)} movies")
    print(f"TF-IDF features shape: {tfidf_features.shape}")

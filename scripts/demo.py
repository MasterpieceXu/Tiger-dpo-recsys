"""
Demo script for testing the trained recommendation model
"""
import os
import sys
import json
import torch
import pandas as pd
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tiger_model import TIGERModel
from config import Config

class RecommendationDemo:
    """Demo class for recommendation system"""
    
    def __init__(self, model_path: str, data_dir: str):
        self.config = Config()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print("Loading TIGER model...")
        self.model = TIGERModel.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Load data mappings
        self.load_mappings(data_dir)
        
        print("Demo ready!")
    
    def load_mappings(self, data_dir: str):
        """Load movie and semantic ID mappings"""
        # Load movies data
        movies_path = os.path.join(data_dir, "processed_movies.csv")
        self.movies_df = pd.read_csv(movies_path)
        
        # Create movie ID to title mapping
        self.movie_id_to_title = dict(zip(
            self.movies_df['movieId'], 
            self.movies_df['title']
        ))
        
        # Load semantic IDs
        semantic_ids_path = os.path.join(data_dir, "item_semantic_ids.jsonl")
        self.semantic_to_movie = {}
        self.movie_to_semantic = {}
        
        with open(semantic_ids_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                movie_id = item['movieId']
                semantic_ids = item['semantic_ids']
                
                # 使用完整的语义 ID 序列作为 key，例如 (a, b)
                self.semantic_to_movie[tuple(semantic_ids)] = movie_id
                
                # Map movie ID to semantic IDs
                self.movie_to_semantic[movie_id] = semantic_ids
        
        print(f"Loaded mappings for {len(self.movie_id_to_title)} movies")
    
    def movie_ids_to_titles(self, movie_ids: List[int]) -> List[str]:
        """Convert movie IDs to titles"""
        titles = []
        for movie_id in movie_ids:
            if movie_id in self.movie_id_to_title:
                titles.append(self.movie_id_to_title[movie_id])
            else:
                titles.append(f"Unknown Movie (ID: {movie_id})")
        return titles
    
    def movie_ids_to_semantic_sequence(self, movie_ids: List[int]) -> List[str]:
        """Convert movie IDs to semantic token sequence"""
        semantic_sequence = []
        for movie_id in movie_ids:
            if movie_id in self.movie_to_semantic:
                semantic_ids = self.movie_to_semantic[movie_id]
                semantic_tokens = [f"<id_{sid}>" for sid in semantic_ids]
                semantic_sequence.extend(semantic_tokens)
            else:
                semantic_sequence.extend(["<unk>", "<unk>"])
        return semantic_sequence
    
    def semantic_tokens_to_movie_ids(self, semantic_tokens: List[str]) -> List[int]:
        """Convert semantic tokens to movie IDs"""
        movie_ids = []
        semantic_ids = []
        for token in semantic_tokens:
            if token.startswith('<id_') and token.endswith('>'):
                try:
                    semantic_ids.append(int(token[4:-1]))
                except ValueError:
                    continue
        if semantic_ids:
            key = tuple(semantic_ids)
            if key in self.semantic_to_movie:
                movie_id = self.semantic_to_movie[key]
                movie_ids.append(movie_id)
        return movie_ids
    
    def recommend_for_user_history(self, movie_ids: List[int], num_recommendations: int = 10) -> Dict:
        """Generate recommendations based on user history"""
        print(f"\nGenerating recommendations for user history...")
        print("User watched:")
        titles = self.movie_ids_to_titles(movie_ids)
        for i, title in enumerate(titles):
            print(f"  {i+1}. {title}")
        
        # Convert to semantic sequence
        semantic_sequence = self.movie_ids_to_semantic_sequence(movie_ids)
        
        # Generate recommendations
        try:
            semantic_recs = self.model.recommend(
                semantic_sequence,
                num_recommendations=num_recommendations * 2,  # Generate more to filter
                num_beams=20
            )
            
            # Convert to movie IDs
            recommended_movie_ids = []
            for semantic_rec in semantic_recs:
                movie_ids_from_rec = self.semantic_tokens_to_movie_ids(
                    [f"<id_{sid}>" for sid in semantic_rec]
                )
                for movie_id in movie_ids_from_rec:
                    if movie_id not in movie_ids and movie_id not in recommended_movie_ids:
                        recommended_movie_ids.append(movie_id)
                        if len(recommended_movie_ids) >= num_recommendations:
                            break
                if len(recommended_movie_ids) >= num_recommendations:
                    break
            
            # Get titles
            recommended_titles = self.movie_ids_to_titles(recommended_movie_ids)
            
            print(f"\nTop {len(recommended_titles)} Recommendations:")
            for i, title in enumerate(recommended_titles):
                print(f"  {i+1}. {title}")
            
            return {
                'user_history': {
                    'movie_ids': movie_ids,
                    'titles': titles
                },
                'recommendations': {
                    'movie_ids': recommended_movie_ids,
                    'titles': recommended_titles
                }
            }
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return None
    
    def search_movies(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for movies by title"""
        query_lower = query.lower()
        matches = []
        
        for movie_id, title in self.movie_id_to_title.items():
            if query_lower in title.lower():
                matches.append({
                    'movie_id': movie_id,
                    'title': title
                })
                if len(matches) >= limit:
                    break
        
        return matches
    
    def interactive_demo(self):
        """Run interactive demo"""
        print("\n" + "=" * 60)
        print("Interactive Movie Recommendation Demo")
        print("=" * 60)
        print("Commands:")
        print("  search <query>     - Search for movies")
        print("  add <movie_id>     - Add movie to history")
        print("  history            - Show current history")
        print("  recommend          - Get recommendations")
        print("  clear              - Clear history")
        print("  quit               - Exit demo")
        print("=" * 60)
        
        user_history = []
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit":
                    break
                elif command.startswith("search "):
                    query = command[7:]
                    matches = self.search_movies(query)
                    if matches:
                        print(f"\nFound {len(matches)} movies:")
                        for match in matches:
                            print(f"  ID: {match['movie_id']} - {match['title']}")
                    else:
                        print("No movies found.")
                
                elif command.startswith("add "):
                    try:
                        movie_id = int(command[4:])
                        if movie_id in self.movie_id_to_title:
                            user_history.append(movie_id)
                            title = self.movie_id_to_title[movie_id]
                            print(f"Added: {title}")
                        else:
                            print("Movie ID not found.")
                    except ValueError:
                        print("Invalid movie ID.")
                
                elif command == "history":
                    if user_history:
                        print("\nCurrent history:")
                        titles = self.movie_ids_to_titles(user_history)
                        for i, title in enumerate(titles):
                            print(f"  {i+1}. {title}")
                    else:
                        print("No movies in history.")
                
                elif command == "recommend":
                    if len(user_history) >= 2:
                        self.recommend_for_user_history(user_history)
                    else:
                        print("Please add at least 2 movies to history first.")
                
                elif command == "clear":
                    user_history = []
                    print("History cleared.")
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nDemo ended. Goodbye!")

def main():
    config = Config()
    
    # Check if model exists
    model_path = os.path.join(config.model_dir, "tiger_final")
    if not os.path.exists(model_path):
        print("Error: Trained model not found!")
        print(f"Please run the training pipeline first.")
        print(f"Expected model path: {model_path}")
        return
    
    # Create demo
    demo = RecommendationDemo(model_path, config.output_dir)
    
    # Example usage
    print("\n" + "=" * 60)
    print("Example Recommendation")
    print("=" * 60)
    
    # Example movie IDs (you can change these)
    example_history = [1, 2, 3]  # Toy Story, Jumanji, Grumpier Old Men
    demo.recommend_for_user_history(example_history)
    
    # Interactive demo
    demo.interactive_demo()

if __name__ == "__main__":
    main()

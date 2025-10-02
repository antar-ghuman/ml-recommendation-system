"""
SVD Collaborative Filtering Algorithm
Matrix factorization using Surprise library
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import pickle
import yaml
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import time


class SVDRecommender:
    """
    Collaborative Filtering using Singular Value Decomposition
    
    This is Netflix Prize-style matrix factorization - learns latent factors
    for users and items to predict ratings.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.train_stats = {}
        
        # For mapping between internal and external IDs
        self.user_map = {}
        self.movie_map = {}
        
    def _load_config(self) -> dict:
        """Load model configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['algorithms']['collaborative_filtering']
    
    def prepare_data(self, train_df: pd.DataFrame) -> Dataset:
        """
        Convert pandas DataFrame to Surprise Dataset format
        
        Args:
            train_df: DataFrame with columns [userId, movieId, rating]
        
        Returns:
            Surprise Dataset object
        """
        # Define rating scale
        reader = Reader(rating_scale=(0.5, 5.0))
        
        # Load from DataFrame
        data = Dataset.load_from_df(
            train_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Build full trainset
        trainset = data.build_full_trainset()
        
        # Store mappings
        self.user_map = {raw_id: inner_id for raw_id, inner_id in trainset._raw2inner_id_users.items()}
        self.movie_map = {raw_id: inner_id for raw_id, inner_id in trainset._raw2inner_id_items.items()}
        
        return trainset
    
    def train(self, train_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Train SVD model
        
        Args:
            train_df: Training data
            verbose: Print training progress
        
        Returns:
            Training statistics
        """
        if verbose:
            print("="*50)
            print("TRAINING SVD MODEL")
            print("="*50)
            print(f"Training samples: {len(train_df):,}")
            print(f"Configuration:")
            print(f"  - Factors: {self.config['n_factors']}")
            print(f"  - Epochs: {self.config['n_epochs']}")
            print(f"  - Learning rate: {self.config['lr_all']}")
            print(f"  - Regularization: {self.config['reg_all']}")
            print()
        
        # Prepare data
        trainset = self.prepare_data(train_df)
        
        # Initialize model
        self.model = SVD(
            n_factors=self.config['n_factors'],
            n_epochs=self.config['n_epochs'],
            lr_all=self.config['lr_all'],
            reg_all=self.config['reg_all'],
            random_state=42,
            verbose=verbose
        )
        
        # Train
        start_time = time.time()
        self.model.fit(trainset)
        training_time = time.time() - start_time
        
        # Store stats
        self.train_stats = {
            'training_time': training_time,
            'n_users': trainset.n_users,
            'n_items': trainset.n_items,
            'n_ratings': trainset.n_ratings,
            'global_mean': trainset.global_mean
        }
        
        if verbose:
            print(f"\n✅ Training complete in {training_time:.2f} seconds")
            print(f"   - Users: {trainset.n_users:,}")
            print(f"   - Movies: {trainset.n_items:,}")
            print(f"   - Global mean rating: {trainset.global_mean:.2f}")
        
        return self.train_stats
    
    def predict(self, user_id: int, movie_id: int) -> Tuple[float, Dict]:
        """
        Predict rating for a user-movie pair
        
        Args:
            user_id: User ID
            movie_id: Movie ID
        
        Returns:
            Tuple of (predicted_rating, details_dict)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make prediction
        pred = self.model.predict(user_id, movie_id)
        
        details = {
            'user_id': user_id,
            'movie_id': movie_id,
            'predicted_rating': pred.est,
            'impossible_to_predict': not pred.details['was_impossible']
        }
        
        return pred.est, details
    
    def recommend_for_user(self, user_id: int, n: int = 10, 
                          exclude_seen: List[int] = None,
                          candidate_movies: List[int] = None) -> pd.DataFrame:
        """
        Generate top-N recommendations for a user
        
        Args:
            user_id: User ID to recommend for
            n: Number of recommendations
            exclude_seen: List of movie IDs user has already seen
            candidate_movies: If provided, only consider these movies
        
        Returns:
            DataFrame with columns [movieId, predicted_rating, rank]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get all movies if not specified
        if candidate_movies is None:
            candidate_movies = list(self.movie_map.keys())
        
        # Remove seen movies
        if exclude_seen:
            candidate_movies = [m for m in candidate_movies if m not in exclude_seen]
        
        # Predict ratings for all candidates
        predictions = []
        for movie_id in candidate_movies:
            pred_rating, _ = self.predict(user_id, movie_id)
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': pred_rating
            })
        
        # Create DataFrame and sort
        recommendations = pd.DataFrame(predictions)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        recommendations = recommendations.head(n)
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        return recommendations.reset_index(drop=True)
    
    def batch_recommend(self, user_ids: List[int], n: int = 10,
                       exclude_seen: Dict[int, List[int]] = None) -> Dict[int, pd.DataFrame]:
        """
        Generate recommendations for multiple users
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            exclude_seen: Dict mapping user_id to list of seen movie_ids
        
        Returns:
            Dict mapping user_id to recommendations DataFrame
        """
        recommendations = {}
        
        for user_id in user_ids:
            seen = exclude_seen.get(user_id, []) if exclude_seen else None
            recs = self.recommend_for_user(user_id, n=n, exclude_seen=seen)
            recommendations[user_id] = recs
        
        return recommendations
    
    def get_similar_items(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """
        Find similar movies using learned item factors
        
        Args:
            movie_id: Movie ID
            n: Number of similar movies to return
        
        Returns:
            DataFrame with similar movies and similarity scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get inner ID for the movie
        if movie_id not in self.movie_map:
            raise ValueError(f"Movie ID {movie_id} not in training data")
        
        inner_id = self.movie_map[movie_id]
        
        # Get item factors
        target_factors = self.model.qi[inner_id]
        
        # Compute similarities with all items
        similarities = []
        for raw_id, inner_id_other in self.movie_map.items():
            if raw_id == movie_id:
                continue
            
            other_factors = self.model.qi[inner_id_other]
            
            # Cosine similarity
            similarity = np.dot(target_factors, other_factors) / (
                np.linalg.norm(target_factors) * np.linalg.norm(other_factors)
            )
            
            similarities.append({
                'movieId': raw_id,
                'similarity': similarity
            })
        
        # Sort and return top N
        similar_items = pd.DataFrame(similarities)
        similar_items = similar_items.sort_values('similarity', ascending=False)
        similar_items = similar_items.head(n)
        similar_items['rank'] = range(1, len(similar_items) + 1)
        
        return similar_items.reset_index(drop=True)
    
    def save_model(self, output_path: str = "models/svd_model.pkl") -> None:
        """Save trained model to disk"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'train_stats': self.train_stats,
            'user_map': self.user_map,
            'movie_map': self.movie_map
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {output_path}")
    
    def load_model(self, model_path: str = "models/svd_model.pkl") -> None:
        """Load trained model from disk"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.train_stats = model_data['train_stats']
        self.user_map = model_data['user_map']
        self.movie_map = model_data['movie_map']
        
        print(f"✅ Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    print("Loading processed data...")
    train_df = pd.read_csv("data/processed/train.csv")
    
    # Initialize and train
    svd = SVDRecommender()
    svd.train(train_df)
    
    # Save model
    svd.save_model()
    
    # Example recommendations
    sample_user = train_df['userId'].iloc[0]
    print(f"\nGenerating recommendations for user {sample_user}...")
    recs = svd.recommend_for_user(sample_user, n=10)
    print(recs)
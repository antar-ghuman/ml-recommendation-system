"""
ALS (Alternating Least Squares) Recommendation Algorithm
Optimized for implicit feedback, parallel training
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import pickle
import yaml
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import time


class ALSRecommender:
    """
    Alternating Least Squares Collaborative Filtering
    
    Better than SVD for:
    - Implicit feedback (views, clicks, watches)
    - Large-scale systems (parallelizable)
    - When you need fast training on sparse data
    
    How it works:
    - Alternates between fixing user factors and solving for item factors
    - Then fixes item factors and solves for user factors
    - Repeats until convergence
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.train_stats = {}
        
        # Mappings
        self.user_map = {}  # raw_id -> internal_id
        self.movie_map = {}  # raw_id -> internal_id
        self.user_map_reverse = {}  # internal_id -> raw_id
        self.movie_map_reverse = {}  # internal_id -> raw_id
        
        # Sparse matrix
        self.user_item_matrix = None
        
    def _load_config(self) -> dict:
        """Load model configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['algorithms']['matrix_factorization']
    
    def prepare_data(self, train_df: pd.DataFrame) -> csr_matrix:
        """
        Convert DataFrame to sparse user-item matrix
        
        For implicit feedback, we convert ratings to confidence scores:
        - High rating = high confidence
        - Low rating = low confidence (but still interaction)
        
        Args:
            train_df: DataFrame with [userId, movieId, rating]
        
        Returns:
            Sparse CSR matrix (users × movies)
        """
        print("Preparing data for ALS...")
        
        # Create user and movie mappings
        unique_users = train_df['userId'].unique()
        unique_movies = train_df['movieId'].unique()
        
        self.user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        
        # Reverse mappings
        self.user_map_reverse = {idx: user_id for user_id, idx in self.user_map.items()}
        self.movie_map_reverse = {idx: movie_id for movie_id, idx in self.movie_map.items()}
        
        # Convert to internal IDs
        train_df = train_df.copy()
        train_df['user_idx'] = train_df['userId'].map(self.user_map)
        train_df['movie_idx'] = train_df['movieId'].map(self.movie_map)
        
        # Convert ratings to confidence scores
        # Option 1: Direct use (treating as implicit)
        # Option 2: Transform (e.g., binary: rating > 3.5 = 1, else 0)
        # We'll use Option 1: Higher rating = higher confidence
        
        # Create sparse matrix
        n_users = len(self.user_map)
        n_movies = len(self.movie_map)
        
        # Build COO format then convert to CSR (efficient for ALS)
        from scipy.sparse import coo_matrix
        
        user_item_matrix = coo_matrix(
            (train_df['rating'].values, 
             (train_df['user_idx'].values, train_df['movie_idx'].values)),
            shape=(n_users, n_movies)
        ).tocsr()
        
        self.user_item_matrix = user_item_matrix
        
        print(f"✅ Sparse matrix created: {n_users:,} users × {n_movies:,} movies")
        print(f"   Density: {user_item_matrix.nnz / (n_users * n_movies) * 100:.4f}%")
        
        return user_item_matrix
    
    def train(self, train_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Train ALS model
        
        Args:
            train_df: Training data
            verbose: Print training progress
        
        Returns:
            Training statistics
        """
        if verbose:
            print("="*50)
            print("TRAINING ALS MODEL")
            print("="*50)
            print(f"Training samples: {len(train_df):,}")
            print(f"Configuration:")
            print(f"  - Factors: {self.config['factors']}")
            print(f"  - Iterations: {self.config['iterations']}")
            print(f"  - Regularization: {self.config['regularization']}")
            print()
        
        # Prepare sparse matrix
        user_item_matrix = self.prepare_data(train_df)
        
        # Initialize ALS model
        self.model = AlternatingLeastSquares(
            factors=self.config['factors'],
            iterations=self.config['iterations'],
            regularization=self.config['regularization'],
            random_state=42,
            use_gpu=False,  # Set to True if you have GPU
            num_threads=0  # 0 = use all available cores
        )
        
        # Train (note: ALS expects items × users, so we transpose)
        start_time = time.time()
        
        if verbose:
            print("Training ALS (this may take 1-2 minutes)...")
        
        # ALS library expects item-user matrix, so we transpose
        self.model.fit(user_item_matrix.T)
        
        training_time = time.time() - start_time
        
        # Store stats
        self.train_stats = {
            'training_time': training_time,
            'n_users': len(self.user_map),
            'n_items': len(self.movie_map),
            'n_ratings': len(train_df),
            'density': user_item_matrix.nnz / (len(self.user_map) * len(self.movie_map))
        }
        
        if verbose:
            print(f"\n✅ Training complete in {training_time:.2f} seconds")
            print(f"   - Users: {len(self.user_map):,}")
            print(f"   - Movies: {len(self.movie_map):,}")
            print(f"   - Ratings: {len(train_df):,}")
            print(f"   - Matrix density: {self.train_stats['density']*100:.4f}%")
        
        return self.train_stats
    
    def predict(self, user_id: int, movie_id: int) -> Tuple[float, Dict]:
        """
        Predict rating for user-movie pair
        
        Args:
            user_id: Raw user ID
            movie_id: Raw movie ID
        
        Returns:
            Tuple of (predicted_score, details_dict)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Check if user/movie in training data
        if user_id not in self.user_map:
            # Cold start: return average score
            return 3.5, {'cold_start': True, 'type': 'user'}
        
        if movie_id not in self.movie_map:
            # Cold start: return average score
            return 3.5, {'cold_start': True, 'type': 'movie'}
        
        # Get internal IDs
        user_idx = self.user_map[user_id]
        movie_idx = self.movie_map[movie_id]
        
        # Predict: dot product of user and item factors
        user_factors = self.model.user_factors[user_idx]
        item_factors = self.model.item_factors[movie_idx]
        
        score = np.dot(user_factors, item_factors)
        
        # Scale to 1-5 range (ALS outputs raw scores)
        # Simple scaling: clip and normalize
        score = np.clip(score, 0, 5)
        
        details = {
            'user_id': user_id,
            'movie_id': movie_id,
            'predicted_score': score,
            'cold_start': False
        }
        
        return score, details
    
    def recommend_for_user(self, user_id: int, n: int = 10,exclude_seen: List[int] = None, filter_already_liked: bool = True) -> pd.DataFrame:
        """Generate top-N recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Check if user exists
        if user_id not in self.user_map:
            print(f"Warning: User {user_id} not in training data. Returning popular items.")
            return self._get_popular_items(n)
        
        user_idx = self.user_map[user_id]
        
        # Get user factors
        user_factors = self.model.user_factors[user_idx]
        
        # Score all items (only items that exist in our mapping)
        n_items = len(self.movie_map)
        item_scores = self.model.item_factors[:n_items] @ user_factors
        
        # Exclude seen items
        if exclude_seen:
            for movie_id in exclude_seen:
                if movie_id in self.movie_map:
                    item_idx = self.movie_map[movie_id]
                    if item_idx < len(item_scores):
                        item_scores[item_idx] = -np.inf
        
        # Get top N items
        top_items_idx = np.argsort(item_scores)[::-1][:n]
        
        # Convert to DataFrame
        recommendations = []
        for rank, item_idx in enumerate(top_items_idx, 1):
            if item_idx in self.movie_map_reverse:
                movie_id = self.movie_map_reverse[item_idx]
                recommendations.append({
                    'movieId': movie_id,
                    'score': float(item_scores[item_idx]),
                    'rank': rank
                })
        
        return pd.DataFrame(recommendations)
    
    def _get_popular_items(self, n: int) -> pd.DataFrame:
        """Fallback: return most popular items"""
        # Get items with most interactions
        item_popularity = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        top_items = item_popularity.argsort()[-n:][::-1]
        
        recommendations = []
        for rank, item_idx in enumerate(top_items, 1):
            movie_id = self.movie_map_reverse[item_idx]
            recommendations.append({
                'movieId': movie_id,
                'score': item_popularity[item_idx],
                'rank': rank
            })
        
        return pd.DataFrame(recommendations)
    
    def get_similar_items(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """
        Find similar movies using learned item factors
        
        Args:
            movie_id: Movie ID
            n: Number of similar movies
        
        Returns:
            DataFrame with similar movies and similarity scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if movie_id not in self.movie_map:
            raise ValueError(f"Movie {movie_id} not in training data")
        
        movie_idx = self.movie_map[movie_id]
        
        # Get similar items from ALS
        similar_items = self.model.similar_items(movie_idx, N=n+1)  # +1 because includes itself
        
        # Convert to DataFrame
        similarities = []
        for item_idx, score in similar_items:
            similar_movie_id = self.movie_map_reverse[item_idx]
            
            # Skip the query movie itself
            if similar_movie_id == movie_id:
                continue
            
            similarities.append({
                'movieId': similar_movie_id,
                'similarity': float(score)
            })
        
        similar_df = pd.DataFrame(similarities[:n])
        similar_df['rank'] = range(1, len(similar_df) + 1)
        
        return similar_df
    
    def save_model(self, output_path: str = "models/als_model.pkl") -> None:
        """Save trained model to disk"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'train_stats': self.train_stats,
            'user_map': self.user_map,
            'movie_map': self.movie_map,
            'user_map_reverse': self.user_map_reverse,
            'movie_map_reverse': self.movie_map_reverse,
            'user_item_matrix': self.user_item_matrix
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {output_path}")
    
    def load_model(self, model_path: str = "models/als_model.pkl") -> None:
        """Load trained model from disk"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.train_stats = model_data['train_stats']
        self.user_map = model_data['user_map']
        self.movie_map = model_data['movie_map']
        self.user_map_reverse = model_data['user_map_reverse']
        self.movie_map_reverse = model_data['movie_map_reverse']
        self.user_item_matrix = model_data['user_item_matrix']
        
        print(f"✅ Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    print("Loading processed data...")
    train_df = pd.read_csv("data/processed/train.csv")
    
    # Initialize and train
    als = ALSRecommender()
    als.train(train_df)
    
    # Save model
    als.save_model()
    
    # Example recommendations
    sample_user = train_df['userId'].iloc[0]
    print(f"\nGenerating recommendations for user {sample_user}...")
    recs = als.recommend_for_user(sample_user, n=10)
    print(recs)
"""
Data Preprocessing Pipeline for Recommendation System
Handles loading, cleaning, feature engineering, and splitting MovieLens data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import pickle
from sklearn.model_selection import train_test_split
from datetime import datetime
import yaml


class DataProcessor:
    """Process MovieLens data for recommendation algorithms"""
    
    def __init__(self, data_path: str = "data/raw/ml-25m", config_path: str = "config/model_config.yaml"):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Will be populated during processing
        self.ratings = None
        self.movies = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.user_stats = None
        self.movie_stats = None
        
    def _load_config(self) -> dict:
        """Load configuration file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> None:
        """Load raw MovieLens data"""
        print("Loading data...")
        
        # Load ratings
        self.ratings = pd.read_csv(self.data_path / "ratings.csv")
        print(f"✅ Loaded {len(self.ratings):,} ratings")
        
        # Load movies
        self.movies = pd.read_csv(self.data_path / "movies.csv")
        print(f"✅ Loaded {len(self.movies):,} movies")
        
        # Load tags (optional - for content-based filtering later)
        self.tags = pd.read_csv(self.data_path / "tags.csv")
        print(f"✅ Loaded {len(self.tags):,} tags")
        
        self._print_data_summary()
    
    def _print_data_summary(self) -> None:
        """Print summary statistics"""
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Users: {self.ratings['userId'].nunique():,}")
        print(f"Movies: {self.ratings['movieId'].nunique():,}")
        print(f"Ratings: {len(self.ratings):,}")
        print(f"Sparsity: {self._calculate_sparsity():.4f}%")
        print(f"Rating range: {self.ratings['rating'].min()} - {self.ratings['rating'].max()}")
        print(f"Date range: {pd.to_datetime(self.ratings['timestamp'], unit='s').min()} to {pd.to_datetime(self.ratings['timestamp'], unit='s').max()}")
        print("="*50 + "\n")
    
    def _calculate_sparsity(self) -> float:
        """Calculate matrix sparsity"""
        n_users = self.ratings['userId'].nunique()
        n_movies = self.ratings['movieId'].nunique()
        n_ratings = len(self.ratings)
        sparsity = (1 - (n_ratings / (n_users * n_movies))) * 100
        return sparsity
    
    def clean_data(self, min_ratings_per_user: int = 20, min_ratings_per_movie: int = 20) -> None:
        """
        Clean data by removing cold-start users/items
        
        Args:
            min_ratings_per_user: Minimum ratings a user must have
            min_ratings_per_movie: Minimum ratings a movie must have
        """
        print("Cleaning data...")
        initial_size = len(self.ratings)
        
        # Remove users with too few ratings
        user_counts = self.ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings_per_user].index
        self.ratings = self.ratings[self.ratings['userId'].isin(valid_users)]
        
        # Remove movies with too few ratings
        movie_counts = self.ratings['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
        self.ratings = self.ratings[self.ratings['movieId'].isin(valid_movies)]
        
        # Filter movies dataframe to match
        self.movies = self.movies[self.movies['movieId'].isin(self.ratings['movieId'].unique())]
        
        removed = initial_size - len(self.ratings)
        print(f"✅ Removed {removed:,} ratings ({removed/initial_size*100:.2f}%)")
        print(f"✅ Remaining: {len(self.ratings):,} ratings")
        print(f"✅ Users: {self.ratings['userId'].nunique():,}")
        print(f"✅ Movies: {self.ratings['movieId'].nunique():,}")
    
    def engineer_features(self) -> None:
        """Create additional features"""
        print("Engineering features...")
        
        # Convert timestamp to datetime
        self.ratings['datetime'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.ratings['year'] = self.ratings['datetime'].dt.year
        self.ratings['month'] = self.ratings['datetime'].dt.month
        self.ratings['day_of_week'] = self.ratings['datetime'].dt.dayofweek
        
        # User statistics
        self.user_stats = self.ratings.groupby('userId').agg({
            'rating': ['mean', 'std', 'count'],
            'movieId': 'nunique'
        }).reset_index()
        self.user_stats.columns = ['userId', 'user_avg_rating', 'user_rating_std', 
                                    'user_rating_count', 'user_unique_movies']
        
        # Movie statistics
        self.movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['mean', 'std', 'count'],
            'userId': 'nunique'
        }).reset_index()
        self.movie_stats.columns = ['movieId', 'movie_avg_rating', 'movie_rating_std', 
                                     'movie_rating_count', 'movie_unique_users']
        
        # Merge stats back to ratings
        self.ratings = self.ratings.merge(self.user_stats, on='userId', how='left')
        self.ratings = self.ratings.merge(self.movie_stats, on='movieId', how='left')
        
        # Parse genres from movies
        self.movies['genres_list'] = self.movies['genres'].str.split('|')
        
        print("✅ Features engineered")
        print(f"   - User stats: avg rating, std, count")
        print(f"   - Movie stats: avg rating, std, count")
        print(f"   - Temporal: year, month, day_of_week")
    
    def split_data(self, test_size: float = None, val_size: float = None, 
                   random_state: int = None, temporal_split: bool = False) -> None:
        """
        Split data into train/val/test
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            temporal_split: If True, split by time instead of random
        """
        print("Splitting data...")
        
        # Get config values if not provided
        if test_size is None:
            test_size = self.config['data_split']['test_size']
        if val_size is None:
            val_size = self.config['data_split']['validation_size']
        if random_state is None:
            random_state = self.config['data_split']['random_state']
        
        if temporal_split:
            # Sort by timestamp
            sorted_ratings = self.ratings.sort_values('timestamp')
            
            # Calculate split indices
            n = len(sorted_ratings)
            test_idx = int(n * (1 - test_size))
            val_idx = int(test_idx * (1 - val_size))
            
            self.train_data = sorted_ratings.iloc[:val_idx].copy()
            self.val_data = sorted_ratings.iloc[val_idx:test_idx].copy()
            self.test_data = sorted_ratings.iloc[test_idx:].copy()
            
            print("✅ Temporal split complete")
        else:
            # Random split
            # First split off test
            train_val, self.test_data = train_test_split(
                self.ratings, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Then split train/val
            val_size_adjusted = val_size / (1 - test_size)
            self.train_data, self.val_data = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=random_state
            )
            
            print("✅ Random split complete")
        
        print(f"   - Train: {len(self.train_data):,} ({len(self.train_data)/len(self.ratings)*100:.1f}%)")
        print(f"   - Val:   {len(self.val_data):,} ({len(self.val_data)/len(self.ratings)*100:.1f}%)")
        print(f"   - Test:  {len(self.test_data):,} ({len(self.test_data)/len(self.ratings)*100:.1f}%)")
    
    def save_processed_data(self, output_dir: str = "data/processed") -> None:
        """Save processed datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Saving processed data...")
        
        # Save splits
        self.train_data.to_csv(output_path / "train.csv", index=False)
        self.val_data.to_csv(output_path / "val.csv", index=False)
        self.test_data.to_csv(output_path / "test.csv", index=False)
        
        # Save metadata
        self.movies.to_csv(output_path / "movies_processed.csv", index=False)
        self.user_stats.to_csv(output_path / "user_stats.csv", index=False)
        self.movie_stats.to_csv(output_path / "movie_stats.csv", index=False)
        
        # Save processing info
        info = {
            'processed_date': datetime.now().isoformat(),
            'total_ratings': len(self.ratings),
            'train_size': len(self.train_data),
            'val_size': len(self.val_data),
            'test_size': len(self.test_data),
            'n_users': self.ratings['userId'].nunique(),
            'n_movies': self.ratings['movieId'].nunique(),
            'sparsity': self._calculate_sparsity()
        }
        
        with open(output_path / "processing_info.yaml", 'w') as f:
            yaml.dump(info, f)
        
        print(f"✅ Data saved to {output_path}")
    
    def get_surprise_format(self, data: pd.DataFrame) -> list:
        """
        Convert to Surprise library format (for SVD algorithm)
        Returns list of (user_id, item_id, rating) tuples
        """
        return [(row['userId'], row['movieId'], row['rating']) 
                for _, row in data.iterrows()]
    
    def run_full_pipeline(self, save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run complete preprocessing pipeline
        
        Returns:
            Dictionary with train/val/test dataframes
        """
        self.load_data()
        self.clean_data()
        self.engineer_features()
        self.split_data()
        
        if save:
            self.save_processed_data()
        
        return {
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data,
            'movies': self.movies,
            'user_stats': self.user_stats,
            'movie_stats': self.movie_stats
        }


if __name__ == "__main__":
    # Run the pipeline
    processor = DataProcessor()
    datasets = processor.run_full_pipeline()
    
    print("\n" + "="*50)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*50)
    print("\nReady to train models!")
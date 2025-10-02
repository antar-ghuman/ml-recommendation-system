"""
Exploration & Debugging Notebook
Use this to test components individually and explore your data

Save this as: notebooks/explore.py
(You can convert to .ipynb later if needed)
"""

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("🔍 RECOMMENDATION SYSTEM EXPLORER")
print("="*60)

# ============================================================
# SECTION 1: CHECK DATA
# ============================================================
print("\n📁 SECTION 1: Data Check")
print("-"*60)

try:
    # Check raw data
    raw_path = Path("../data/raw/ml-25m")
    if raw_path.exists():
        ratings = pd.read_csv(raw_path / "ratings.csv")
        movies = pd.read_csv(raw_path / "movies.csv")
        print(f"✅ Raw data loaded")
        print(f"   Ratings: {len(ratings):,} rows")
        print(f"   Movies: {len(movies):,} rows")
        
        # Quick stats
        print(f"\n📊 Quick Stats:")
        print(f"   Users: {ratings['userId'].nunique():,}")
        print(f"   Movies: {ratings['movieId'].nunique():,}")
        print(f"   Avg rating: {ratings['rating'].mean():.2f}")
        print(f"   Rating std: {ratings['rating'].std():.2f}")
    else:
        print("❌ Raw data not found. Run: python scripts/download_data.py")
except Exception as e:
    print(f"❌ Error loading raw data: {e}")

try:
    # Check processed data
    proc_path = Path("../data/processed")
    if (proc_path / "train.csv").exists():
        train = pd.read_csv(proc_path / "train.csv")
        val = pd.read_csv(proc_path / "val.csv")
        test = pd.read_csv(proc_path / "test.csv")
        print(f"\n✅ Processed data loaded")
        print(f"   Train: {len(train):,} rows")
        print(f"   Val: {len(val):,} rows")
        print(f"   Test: {len(test):,} rows")
    else:
        print("\n⚠️  Processed data not found. Run: python run_pipeline.py")
except Exception as e:
    print(f"❌ Error loading processed data: {e}")

# ============================================================
# SECTION 2: TEST MODEL LOADING
# ============================================================
print("\n\n🤖 SECTION 2: Model Check")
print("-"*60)

try:
    from src.algorithms.svd_algorithm import SVDRecommender
    
    model_path = Path("../models/svd_model.pkl")
    if model_path.exists():
        svd = SVDRecommender()
        svd.load_model(str(model_path))
        print("✅ Model loaded successfully")
        print(f"   Training stats: {svd.train_stats}")
    else:
        print("⚠️  Model not found. Train it first: python run_pipeline.py")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ============================================================
# SECTION 3: GENERATE TEST RECOMMENDATIONS
# ============================================================
print("\n\n🎬 SECTION 3: Test Recommendations")
print("-"*60)

try:
    if 'svd' in locals() and 'train' in locals() and 'movies' in locals():
        # Pick a random user
        test_user = train['userId'].sample(1).iloc[0]
        
        # Get their history
        user_history = train[train['userId'] == test_user].merge(
            movies[['movieId', 'title', 'genres']], 
            on='movieId'
        ).sort_values('rating', ascending=False)
        
        print(f"User {test_user}'s Top Rated Movies:")
        print(user_history[['title', 'rating', 'genres']].head(5).to_string(index=False))
        
        # Generate recommendations
        seen_movies = user_history['movieId'].tolist()
        recs = svd.recommend_for_user(test_user, n=10, exclude_seen=seen_movies)
        recs = recs.merge(movies[['movieId', 'title', 'genres']], on='movieId')
        
        print(f"\n🎯 Top 10 Recommendations:")
        print(recs[['rank', 'title', 'predicted_rating', 'genres']].to_string(index=False))
        
    else:
        print("⚠️  Need model and data loaded first")
except Exception as e:
    print(f"❌ Error generating recommendations: {e}")

# ============================================================
# SECTION 4: EVALUATE MODEL
# ============================================================
print("\n\n📊 SECTION 4: Quick Evaluation")
print("-"*60)

try:
    if 'svd' in locals() and 'test' in locals():
        from src.evaluation.evaluation_metrics import RecommendationEvaluator
        
        evaluator = RecommendationEvaluator()
        
        # Sample evaluation (faster)
        test_sample = test.sample(min(10000, len(test)))
        
        print("Evaluating on test sample...")
        
        # Accuracy metrics
        true_ratings = []
        pred_ratings = []
        
        for _, row in test_sample.iterrows():
            try:
                pred, _ = svd.predict(row['userId'], row['movieId'])
                true_ratings.append(row['rating'])
                pred_ratings.append(pred)
            except:
                continue
        
        if len(true_ratings) > 0:
            rmse = evaluator.rmse(true_ratings, pred_ratings)
            mae = evaluator.mae(true_ratings, pred_ratings)
            
            print(f"✅ RMSE: {rmse:.4f}")
            print(f"✅ MAE: {mae:.4f}")
        
    else:
        print("⚠️  Need model and test data loaded")
except Exception as e:
    print(f"❌ Error during evaluation: {e}")

# ============================================================
# SECTION 5: DATA VISUALIZATIONS (OPTIONAL)
# ============================================================
print("\n\n📈 SECTION 5: Visualizations")
print("-"*60)
print("(Skipping plots - uncomment code to visualize)")

# Uncomment to create visualizations:
"""
if 'ratings' in locals():
    # Rating distribution
    plt.figure(figsize=(10, 5))
    ratings['rating'].hist(bins=10, edgecolor='black')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('notebooks/rating_distribution.png')
    print("✅ Saved: rating_distribution.png")
    
    # Ratings per user
    plt.figure(figsize=(10, 5))
    user_counts = ratings.groupby('userId').size()
    user_counts.hist(bins=50, edgecolor='black')
    plt.title('Ratings per User Distribution')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.yscale('log')
    plt.savefig('notebooks/user_activity.png')
    print("✅ Saved: user_activity.png")
"""

# ============================================================
# SECTION 6: DIAGNOSTIC TOOLS
# ============================================================
print("\n\n🔧 SECTION 6: Diagnostic Tools")
print("-"*60)

def check_cold_start_users(df, min_ratings=5):
    """Find users with few ratings (cold start problem)"""
    user_counts = df.groupby('userId').size()
    cold_start = user_counts[user_counts < min_ratings]
    return cold_start

def check_cold_start_items(df, min_ratings=5):
    """Find items with few ratings (cold start problem)"""
    item_counts = df.groupby('movieId').size()
    cold_start = item_counts[item_counts < min_ratings]
    return cold_start

try:
    if 'train' in locals():
        cold_users = check_cold_start_users(train)
        cold_items = check_cold_start_items(train)
        
        print(f"Cold start analysis:")
        print(f"   Users with <5 ratings: {len(cold_users):,} ({len(cold_users)/train['userId'].nunique()*100:.1f}%)")
        print(f"   Movies with <5 ratings: {len(cold_items):,} ({len(cold_items)/train['movieId'].nunique()*100:.1f}%)")
except Exception as e:
    print(f"❌ Error in diagnostic: {e}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
print("\n\n💡 HELPER FUNCTIONS")
print("-"*60)
print("Available functions:")
print("  - check_cold_start_users(df, min_ratings=5)")
print("  - check_cold_start_items(df, min_ratings=5)")
print("\nExample usage:")
print("  cold_users = check_cold_start_users(train, min_ratings=10)")
print("  print(f'Found {len(cold_users)} cold start users')")

print("\n" + "="*60)
print("✅ EXPLORATION COMPLETE")
print("="*60)
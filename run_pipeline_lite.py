"""
Main Pipeline: Train and Evaluate SVD Recommendation Model (LITE VERSION)
This version uses a sample of data to avoid memory issues
"""

import sys
from pathlib import Path
import pandas as pd
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processor import DataProcessor
from src.algorithms.svd_algorithm import SVDRecommender
from src.evaluation.evaluation_metrics import RecommendationEvaluator


def main():
    """Run the complete training pipeline with memory optimizations"""
    
    print("\n" + "="*70)
    print("MULTI-ALGORITHM RECOMMENDATION SYSTEM - TRAINING PIPELINE (LITE)")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # ========== STEP 1: DATA PREPROCESSING ==========
    print("STEP 1/4: Data Preprocessing")
    print("-" * 70)
    
    processor = DataProcessor()
    
    # Load data
    processor.load_data()
    
    # MEMORY OPTIMIZATION: Sample 20% of data for training
    print("\n🔧 Memory optimization: Using 20% sample of data")
    processor.ratings = processor.ratings.sample(frac=0.2, random_state=42)
    print(f"✅ Sampled to {len(processor.ratings):,} ratings\n")
    
    # Clean with stricter thresholds to reduce size
    processor.clean_data(min_ratings_per_user=30, min_ratings_per_movie=30)
    
    processor.engineer_features()
    processor.split_data()
    processor.save_processed_data()
    
    train_df = processor.train_data
    val_df = processor.val_data
    test_df = processor.test_data
    movies_df = processor.movies
    
    print(f"✅ Data preprocessing complete")
    print(f"   Training set: {len(train_df):,} ratings")
    print(f"   Validation set: {len(val_df):,} ratings")
    print(f"   Test set: {len(test_df):,} ratings\n")
    
    # ========== STEP 2: TRAIN SVD MODEL ==========
    print("STEP 2/4: Training SVD Model")
    print("-" * 70)
    
    svd = SVDRecommender()
    svd.train(train_df, verbose=True)
    
    # Save model
    svd.save_model("models/svd_model.pkl")
    print()
    
    # ========== STEP 3: EVALUATE ON TEST SET ==========
    print("STEP 3/4: Model Evaluation")
    print("-" * 70)
    
    evaluator = RecommendationEvaluator()
    
    # Evaluate on smaller validation sample
    print("\n📊 Validation Set Performance (1000 samples):")
    val_sample = val_df.sample(min(1000, len(val_df)))
    
    # Accuracy metrics only (faster)
    true_ratings = []
    pred_ratings = []
    
    for _, row in val_sample.iterrows():
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
    
    # Quick ranking metrics on 100 users
    print("\n📊 Ranking Metrics (100 users):")
    sample_users = val_df['userId'].unique()[:100]
    user_interactions = val_df.groupby('userId')['movieId'].apply(list).to_dict()
    
    precisions = []
    for user_id in sample_users:
        try:
            recs = svd.recommend_for_user(user_id, n=10)
            recommended = recs['movieId'].tolist()
            relevant = user_interactions.get(user_id, [])
            
            if len(relevant) > 0:
                precision = len(set(recommended) & set(relevant)) / 10
                precisions.append(precision)
        except:
            continue
    
    if precisions:
        print(f"✅ Precision@10: {sum(precisions)/len(precisions):.4f}")
    
    print()
    
    # ========== STEP 4: GENERATE SAMPLE RECOMMENDATIONS ==========
    print("STEP 4/4: Sample Recommendations")
    print("-" * 70)
    
    # Get a random user
    sample_user_id = train_df['userId'].sample(1).iloc[0]
    
    # Get movies they've rated
    user_movies = train_df[train_df['userId'] == sample_user_id]['movieId'].tolist()
    
    # Generate recommendations
    print(f"\nGenerating recommendations for User {sample_user_id}")
    print(f"This user has rated {len(user_movies)} movies")
    
    recommendations = svd.recommend_for_user(
        sample_user_id, 
        n=10, 
        exclude_seen=user_movies
    )
    
    # Merge with movie info
    recommendations = recommendations.merge(
        movies_df[['movieId', 'title', 'genres']], 
        on='movieId', 
        how='left'
    )
    
    print("\nTop 10 Recommendations:")
    print(recommendations[['rank', 'title', 'predicted_rating', 'genres']].to_string(index=False))
    
    # ========== SUMMARY ==========
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\n📁 Outputs saved:")
    print(f"   - Processed data: data/processed/")
    print(f"   - Trained model: models/svd_model.pkl")
    print(f"\n📊 Key Metrics:")
    if len(true_ratings) > 0:
        print(f"   - RMSE: {rmse:.4f}")
        print(f"   - MAE: {mae:.4f}")
    if precisions:
        print(f"   - Precision@10: {sum(precisions)/len(precisions):.4f}")
    
    print("\n✨ Next Steps:")
    print("   1. Try different hyperparameters in config/model_config.yaml")
    print("   2. Build additional algorithms (ALS, Neural CF, Content-Based)")
    print("   3. Set up A/B testing experiments")
    print("   4. Build FastAPI service for real-time recommendations")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
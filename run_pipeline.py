"""
Main Pipeline: Train and Evaluate SVD Recommendation Model
Run this script to execute the full pipeline
"""

import sys
from pathlib import Path
import pandas as pd
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules (these will be saved in src/)
from src.data_processor import DataProcessor
from src.algorithms.svd_algorithm import SVDRecommender
from src.evaluation.evaluation_metrics import RecommendationEvaluator


def main():
    """Run the complete training pipeline"""
    
    print("\n" + "="*70)
    print("MULTI-ALGORITHM RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # ========== STEP 1: DATA PREPROCESSING ==========
    print("STEP 1/4: Data Preprocessing")
    print("-" * 70)
    
    processor = DataProcessor()
    datasets = processor.run_full_pipeline()
    
    train_df = datasets['train']
    val_df = datasets['val']
    test_df = datasets['test']
    movies_df = datasets['movies']
    
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
    
    # Evaluate on validation set first
    print("\n📊 Validation Set Performance:")
    val_results = evaluator.evaluate_model(svd, val_df, k_values=[5, 10, 20])
    
    # Evaluate on test set
    print("\n📊 Test Set Performance:")
    test_results = evaluator.evaluate_model(svd, test_df, k_values=[5, 10, 20])
    
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
    print(f"   - RMSE: {test_results.get('rmse', 'N/A')}")
    print(f"   - Precision@10: {test_results.get('precision@10', 'N/A'):.4f}")
    print(f"   - NDCG@10: {test_results.get('ndcg@10', 'N/A'):.4f}")
    
    print("\n✨ Next Steps:")
    print("   1. Try different hyperparameters in config/model_config.yaml")
    print("   2. Build additional algorithms (ALS, Neural CF, Content-Based)")
    print("   3. Set up A/B testing experiments")
    print("   4. Build FastAPI service for real-time recommendations")
    print("="*70 + "\n")
    
    return {
        'test_results': test_results,
        'val_results': val_results,
        'training_time': svd.train_stats['training_time']
    }


if __name__ == "__main__":
    results = main()
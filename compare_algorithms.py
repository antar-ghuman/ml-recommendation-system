"""
Compare SVD and ALS Algorithms
Run head-to-head comparison with MLflow tracking
"""

import sys
from pathlib import Path
import pandas as pd
import time
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.algorithms.svd_algorithm import SVDRecommender
from src.algorithms.als_algorithm import ALSRecommender
from src.evaluation.evaluation_metrics import RecommendationEvaluator
from src.experiment_tracker import ExperimentTracker, track_model_training


def load_config():
    """Load model configuration"""
    with open("config/model_config.yaml", 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run SVD vs ALS comparison"""
    
    print("\n" + "="*70)
    print("SVD vs ALS ALGORITHM COMPARISON")
    print("="*70 + "\n")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_name="svd_vs_als_comparison")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    print(f"✅ Data loaded:")
    print(f"   Train: {len(train_df):,} ratings")
    print(f"   Val: {len(val_df):,} ratings")
    print(f"   Test: {len(test_df):,} ratings\n")
    
    # Load config
    config = load_config()
    
    results = {}
    
    # ========== ALGORITHM 1: SVD ==========
    print("\n" + "🔵"*35)
    print("ALGORITHM 1: SVD (Singular Value Decomposition)")
    print("🔵"*35 + "\n")
    
    svd = SVDRecommender()
    svd_params = config['algorithms']['collaborative_filtering']
    
    svd_results = track_model_training(
        tracker=tracker,
        run_name="svd_baseline",
        model=svd,
        train_func=svd.train,
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        algorithm_name="SVD",
        hyperparams=svd_params
    )
    
    results['SVD'] = svd_results
    
    # ========== ALGORITHM 2: ALS ==========
    print("\n" + "🟢"*35)
    print("ALGORITHM 2: ALS (Alternating Least Squares)")
    print("🟢"*35 + "\n")
    
    als = ALSRecommender()
    als_params = config['algorithms']['matrix_factorization']
    
    als_results = track_model_training(
        tracker=tracker,
        run_name="als_baseline",
        model=als,
        train_func=als.train,
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        algorithm_name="ALS",
        hyperparams=als_params
    )
    
    results['ALS'] = als_results
    
    # ========== SAMPLE RECOMMENDATIONS ==========
    print("\n" + "="*70)
    print("📋 SAMPLE RECOMMENDATIONS COMPARISON")
    print("="*70)
    
    # Pick a user that exists in both models
    svd_users = set(svd.user_map.keys())
    als_users = set(als.user_map.keys())
    common_users = svd_users & als_users
    
    if len(common_users) > 0:
        sample_user = list(common_users)[0]
        user_movies = train_df[train_df['userId'] == sample_user]['movieId'].tolist()
        
        print(f"\nUser {sample_user} (has rated {len(user_movies)} movies)")
        
        print("\n🔵 SVD Top 5 Recommendations:")
        svd_recs = svd.recommend_for_user(sample_user, n=5, exclude_seen=user_movies)
        if 'predicted_rating' in svd_recs.columns:
            print(svd_recs[['rank', 'movieId', 'predicted_rating']].to_string(index=False))
        else:
            print(svd_recs)
        
        print("\n🟢 ALS Top 5 Recommendations:")
        als_recs = als.recommend_for_user(sample_user, n=5, exclude_seen=user_movies)
        if 'score' in als_recs.columns:
            print(als_recs[['rank', 'movieId', 'score']].to_string(index=False))
        else:
            print(als_recs)
    else:
        print("\n⚠️  No common users between models")
    
    # ========== COMPARISON ==========
    print("\n" + "="*70)
    print("📊 RESULTS COMPARISON")
    print("="*70 + "\n")
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.to_string())
    
    # Determine winner
    print("\n" + "="*70)
    print("🏆 WINNER DETERMINATION")
    print("="*70 + "\n")
    
    if results['SVD']['test_rmse'] < results['ALS']['test_rmse']:
        winner = "SVD"
        improvement = ((results['ALS']['test_rmse'] - results['SVD']['test_rmse']) / 
                      results['ALS']['test_rmse'] * 100)
        print(f"🏆 Winner: SVD")
        print(f"   SVD has {improvement:.2f}% better RMSE than ALS")
    else:
        winner = "ALS"
        improvement = ((results['SVD']['test_rmse'] - results['ALS']['test_rmse']) / 
                      results['SVD']['test_rmse'] * 100)
        print(f"🏆 Winner: ALS")
        print(f"   ALS has {improvement:.2f}% better RMSE than SVD")
    
    # Statistical significance test
    print("\n📈 Statistical Significance:")
    evaluator = RecommendationEvaluator()
    
    # Get predictions for same test sample
    test_sample = test_df.sample(min(1000, len(test_df)), random_state=42)
    
    svd_predictions = []
    als_predictions = []
    true_values = []
    
    for _, row in test_sample.iterrows():
        try:
            svd_pred, _ = svd.predict(row['userId'], row['movieId'])
            als_pred, _ = als.predict(row['userId'], row['movieId'])
            
            svd_predictions.append(svd_pred)
            als_predictions.append(als_pred)
            true_values.append(row['rating'])
        except:
            continue
    
    # Calculate errors for each prediction
    svd_errors = [abs(true - pred) for true, pred in zip(true_values, svd_predictions)]
    als_errors = [abs(true - pred) for true, pred in zip(true_values, als_predictions)]
    
    # A/B test analysis
    ab_results = evaluator.ab_test_analysis(
        control_metrics=svd_errors,
        treatment_metrics=als_errors,
        metric_name="Absolute Error (lower is better)"
    )
    
    evaluator.print_ab_results(ab_results)
    
    # ========== RECOMMENDATIONS ==========
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70 + "\n")
    
    print(f"Based on this comparison:")
    print(f"1. {winner} performs better on this dataset")
    print(f"2. RMSE difference: {abs(results['SVD']['test_rmse'] - results['ALS']['test_rmse']):.4f}")
    print(f"3. Training time - SVD: {results['SVD']['training_time']:.1f}s, ALS: {results['ALS']['training_time']:.1f}s")
    
    if ab_results['is_significant']:
        print(f"\n✅ The difference IS statistically significant (p={ab_results['p_value']:.4f})")
        print(f"   Recommendation: Use {winner} in production")
    else:
        print(f"\n⚠️  The difference is NOT statistically significant (p={ab_results['p_value']:.4f})")
        print(f"   Recommendation: Either algorithm is fine, choose based on other factors:")
        print(f"   - Training speed: {'SVD' if results['SVD']['training_time'] < results['ALS']['training_time'] else 'ALS'}")
        print(f"   - Implicit feedback support: ALS")
        print(f"   - Explicit ratings: SVD")
    
    # ========== MLFLOW UI ==========
    print("\n" + "="*70)
    print("📊 VIEW DETAILED RESULTS IN MLFLOW")
    print("="*70)
    print("\nTo view interactive comparison:")
    print("  1. Run: mlflow ui")
    print("  2. Open: http://localhost:5000")
    print("  3. Compare runs side-by-side")
    print("  4. Visualize metrics over time")
    print("="*70 + "\n")
    
    # Save comparison summary
    summary_path = Path("experiments/svd_vs_als_summary.csv")
    summary_path.parent.mkdir(exist_ok=True)
    comparison_df.to_csv(summary_path)
    print(f"✅ Comparison summary saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    results = main()
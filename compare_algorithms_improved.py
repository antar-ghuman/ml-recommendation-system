"""
Compare SVD vs ALS with Proper Implicit Feedback Handling
This version converts explicit ratings to implicit signals for ALS
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent))

from src.algorithms.svd_algorithm import SVDRecommender
from src.algorithms.als_algorithm import ALSRecommender
from src.evaluation.evaluation_metrics import RecommendationEvaluator
from src.experiment_tracker import ExperimentTracker, track_model_training


def convert_to_implicit(train_df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    Convert explicit ratings to implicit feedback
    
    Strategy 1: Binary conversion
    - rating >= threshold → 1.0 (liked/watched)
    - rating < threshold → 0.0 (didn't like/skipped)
    
    Strategy 2: Confidence scores
    - High ratings → high confidence
    - Low ratings → low confidence
    
    Args:
        train_df: DataFrame with explicit ratings
        threshold: Rating threshold for binary conversion
    
    Returns:
        DataFrame with implicit confidence scores
    """
    df = train_df.copy()
    
    # Strategy: Convert ratings to confidence
    # rating 5.0 → confidence 5
    # rating 4.5 → confidence 4
    # rating 4.0 → confidence 3
    # rating < 4.0 → confidence 1 (still an interaction, low confidence)
    
    df['confidence'] = df['rating'].apply(lambda x: 
        5 if x >= 4.5 else
        4 if x >= 4.0 else
        3 if x >= 3.5 else
        2 if x >= 3.0 else
        1
    )
    
    # Replace rating with confidence for ALS
    df['rating'] = df['confidence']
    
    return df


def main():
    print("\n" + "="*70)
    print("SVD vs ALS COMPARISON (IMPROVED)")
    print("With Proper Implicit Feedback Handling for ALS")
    print("="*70 + "\n")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_name="svd_vs_als_improved")
    
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
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    results = {}
    
    # ========== ALGORITHM 1: SVD (Explicit Ratings) ==========
    print("\n" + "🔵"*35)
    print("ALGORITHM 1: SVD - Explicit Ratings (1-5 stars)")
    print("🔵"*35 + "\n")
    
    svd = SVDRecommender()
    svd_params = config['algorithms']['collaborative_filtering']
    
    svd_results = track_model_training(
        tracker=tracker,
        run_name="svd_explicit",
        model=svd,
        train_func=svd.train,
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        algorithm_name="SVD",
        hyperparams=svd_params
    )
    
    results['SVD_Explicit'] = svd_results
    
    # ========== ALGORITHM 2: ALS (Implicit Feedback) ==========
    print("\n" + "🟢"*35)
    print("ALGORITHM 2: ALS - Implicit Feedback (confidence scores)")
    print("🟢"*35 + "\n")
    
    # Convert to implicit feedback
    print("Converting ratings to implicit feedback...")
    train_implicit = convert_to_implicit(train_df)
    val_implicit = convert_to_implicit(val_df)
    test_implicit = convert_to_implicit(test_df)
    
    print("✅ Conversion complete:")
    print("   Rating 5.0 → Confidence 5")
    print("   Rating 4.5 → Confidence 4")
    print("   Rating 4.0 → Confidence 3")
    print("   Rating 3.5 → Confidence 2")
    print("   Rating <3.5 → Confidence 1\n")
    
    als = ALSRecommender()
    als_params = config['algorithms']['matrix_factorization']
    
    als_results = track_model_training(
        tracker=tracker,
        run_name="als_implicit",
        model=als,
        train_func=als.train,
        train_data=train_implicit,
        val_data=val_implicit,
        test_data=test_implicit,
        algorithm_name="ALS",
        hyperparams=als_params
    )
    
    results['ALS_Implicit'] = als_results
    
    # ========== COMPARISON ==========
    print("\n" + "="*70)
    print("📊 RESULTS COMPARISON")
    print("="*70 + "\n")
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.to_string())
    
    # ========== ANALYSIS ==========
    print("\n" + "="*70)
    print("🔍 ANALYSIS")
    print("="*70 + "\n")
    
    print("Key Insights:")
    print(f"1. SVD (explicit): RMSE = {results['SVD_Explicit']['test_rmse']:.4f}")
    print(f"2. ALS (implicit): RMSE = {results['ALS_Implicit']['test_rmse']:.4f}")
    print(f"\n3. Training Time:")
    print(f"   - SVD: {results['SVD_Explicit']['training_time']:.2f}s")
    print(f"   - ALS: {results['ALS_Implicit']['training_time']:.2f}s")
    
    improvement = ((results['ALS_Implicit']['test_rmse'] - results['SVD_Explicit']['test_rmse']) / 
                   results['ALS_Implicit']['test_rmse'] * 100)
    
    if results['SVD_Explicit']['test_rmse'] < results['ALS_Implicit']['test_rmse']:
        print(f"\n🏆 Winner: SVD (better by {improvement:.1f}%)")
        print("\nWhy SVD wins:")
        print("  - Optimized for explicit ratings (1-5 stars)")
        print("  - Our dataset has explicit ratings")
        print("  - Direct rating prediction")
    else:
        print(f"\n🏆 Winner: ALS (better by {abs(improvement):.1f}%)")
        print("\nWhy ALS wins:")
        print("  - Better at modeling confidence/preference strength")
        print("  - Robust to rating scale differences")
    
    print("\n" + "="*70)
    print("💡 PRODUCTION RECOMMENDATIONS")
    print("="*70 + "\n")
    
    print("For Netflix-style system:")
    print("1. Use SVD for:")
    print("   - Predicting star ratings")
    print("   - 'Rate this movie' features")
    print("   - User preference surveys")
    print("\n2. Use ALS for:")
    print("   - Watch completion prediction")
    print("   - Click-through rate")
    print("   - 'Continue watching' recommendations")
    print("   - Implicit engagement signals")
    print("\n3. Best approach: HYBRID")
    print("   - Blend both: 60% SVD + 40% ALS")
    print("   - Use SVD for cold-start with explicit ratings")
    print("   - Use ALS for mature users with lots of watch history")
    
    # Save results
    summary_path = Path("experiments/svd_vs_als_improved_summary.csv")
    summary_path.parent.mkdir(exist_ok=True)
    comparison_df.to_csv(summary_path)
    print(f"\n✅ Results saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    results = main()
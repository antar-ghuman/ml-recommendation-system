"""
Lightweight Hyperparameter Tuning for SVD
Faster version with fewer combinations
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).parent))

from src.algorithms.svd_algorithm import SVDRecommender
from src.evaluation.evaluation_metrics import RecommendationEvaluator
from src.experiment_tracker import ExperimentTracker


def main():
    print("\n" + "="*70)
    print("SVD HYPERPARAMETER TUNING (LITE)")
    print("="*70 + "\n")
    
    tracker = ExperimentTracker(experiment_name="svd_tuning_lite")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    print(f"✅ Data loaded\n")
    
    # Smaller, focused grid (9 combinations instead of 81)
    configs = [
        # Baseline
        {'n_factors': 50, 'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.02, 'name': 'baseline'},
        
        # More factors
        {'n_factors': 100, 'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.02, 'name': 'more_factors'},
        {'n_factors': 150, 'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.02, 'name': 'many_factors'},
        
        # More epochs
        {'n_factors': 50, 'n_epochs': 15, 'lr_all': 0.005, 'reg_all': 0.02, 'name': 'more_epochs'},
        {'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02, 'name': 'many_epochs'},
        
        # Learning rate variations
        {'n_factors': 50, 'n_epochs': 10, 'lr_all': 0.007, 'reg_all': 0.02, 'name': 'higher_lr'},
        
        # Regularization variations
        {'n_factors': 50, 'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.01, 'name': 'less_reg'},
        {'n_factors': 50, 'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.05, 'name': 'more_reg'},
        
        # Best combo guess
        {'n_factors': 100, 'n_epochs': 15, 'lr_all': 0.007, 'reg_all': 0.02, 'name': 'optimized'},
    ]
    
    print(f"Testing {len(configs)} configurations\n")
    print("="*70 + "\n")
    
    results = []
    evaluator = RecommendationEvaluator()
    
    for idx, config in enumerate(configs, 1):
        print(f"[{idx}/{len(configs)}] {config['name']}")
        print(f"  factors={config['n_factors']}, epochs={config['n_epochs']}, lr={config['lr_all']}, reg={config['reg_all']}")
        
        with tracker.start_run(run_name=config['name'], tags={'experiment': 'tuning'}):
            # Log params
            tracker.log_params(config)
            
            # Train
            svd = SVDRecommender()
            svd.config = {k: v for k, v in config.items() if k != 'name'}
            
            train_stats = svd.train(train_df, verbose=False)
            
            # Quick eval on validation
            val_sample = val_df.sample(min(500, len(val_df)), random_state=42)
            
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
                val_rmse = evaluator.rmse(true_ratings, pred_ratings)
                val_mae = evaluator.mae(true_ratings, pred_ratings)
                
                tracker.log_metrics({
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'training_time': train_stats['training_time']
                })
                
                results.append({
                    'name': config['name'],
                    'n_factors': config['n_factors'],
                    'n_epochs': config['n_epochs'],
                    'lr_all': config['lr_all'],
                    'reg_all': config['reg_all'],
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'training_time': train_stats['training_time']
                })
                
                print(f"  ✅ RMSE: {val_rmse:.4f}, Time: {train_stats['training_time']:.2f}s\n")
    
    # Results
    results_df = pd.DataFrame(results).sort_values('val_rmse')
    
    print("\n" + "="*70)
    print("🏆 RESULTS (sorted by RMSE)")
    print("="*70 + "\n")
    print(results_df.to_string(index=False))
    
    best = results_df.iloc[0]
    
    print("\n" + "="*70)
    print("✨ BEST CONFIGURATION")
    print("="*70)
    print(f"Name: {best['name']}")
    print(f"Factors: {int(best['n_factors'])}")
    print(f"Epochs: {int(best['n_epochs'])}")
    print(f"Learning Rate: {best['lr_all']}")
    print(f"Regularization: {best['reg_all']}")
    print(f"\nValidation RMSE: {best['val_rmse']:.4f}")
    print(f"Training Time: {best['training_time']:.2f}s")
    
    # Save
    results_path = Path("experiments/hyperparameter_tuning_results.csv")
    results_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    # Save best config
    best_config = {
        'algorithms': {
            'collaborative_filtering': {
                'name': 'SVD',
                'n_factors': int(best['n_factors']),
                'n_epochs': int(best['n_epochs']),
                'lr_all': float(best['lr_all']),
                'reg_all': float(best['reg_all'])
            }
        }
    }
    
    config_path = Path("config/model_config_optimized.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"✅ Best config saved to: {config_path}")
    print("\n" + "="*70)
    
    return results_df


if __name__ == "__main__":
    results = main()
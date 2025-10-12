"""
Experiment Tracker using MLflow
Tracks model training, hyperparameters, and metrics
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import json


class ExperimentTracker:
    """
    Wrapper around MLflow for tracking recommendation system experiments
    """
    
    def __init__(self, experiment_name: str = "recommendation_system"):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment (groups related runs)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI (local file storage)
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        print(f"✅ Experiment tracker initialized: {experiment_name}")
        print(f"   Experiment ID: {self.experiment_id}")
    
    def start_run(self, run_name: str, tags: Dict[str, str] = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for this run (e.g., "svd_baseline", "als_v1")
            tags: Optional tags for organizing runs
        """
        tags = tags or {}
        tags['run_name'] = run_name
        
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters
        
        Args:
            params: Dictionary of parameter name -> value
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (for tracking over time)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, artifact_path: str = "model"):
        """
        Log model artifact
        
        Args:
            model: Trained model object
            artifact_path: Path within run's artifact directory
        """
        import pickle
        
        # Save model locally first
        temp_path = Path("temp_model.pkl")
        with open(temp_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log to MLflow
        mlflow.log_artifact(str(temp_path), artifact_path)
        
        # Cleanup
        temp_path.unlink()
    
    def log_dataframe(self, df: pd.DataFrame, filename: str):
        """
        Log a DataFrame as CSV artifact
        
        Args:
            df: DataFrame to log
            filename: Name for the CSV file
        """
        temp_path = Path(f"temp_{filename}")
        df.to_csv(temp_path, index=False)
        mlflow.log_artifact(str(temp_path))
        temp_path.unlink()
    
    def log_dict(self, data: Dict, filename: str):
        """
        Log dictionary as JSON artifact
        
        Args:
            data: Dictionary to log
            filename: Name for JSON file
        """
        temp_path = Path(f"temp_{filename}")
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(str(temp_path))
        temp_path.unlink()
    
    def end_run(self):
        """End the current run"""
        mlflow.end_run()
    
    def compare_runs(self, metric_name: str = "rmse", top_n: int = 5) -> pd.DataFrame:
        """
        Compare runs by a specific metric
        
        Args:
            metric_name: Metric to compare (e.g., "rmse", "mae")
            top_n: Number of top runs to return
        
        Returns:
            DataFrame with run comparisons
        """
        # Search all runs in this experiment
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} ASC"]
        )
        
        if len(runs) == 0:
            print("No runs found in this experiment")
            return pd.DataFrame()
        
        # Select relevant columns
        columns_to_show = ['run_id', 'start_time', 'status']
        
        # Add all metrics
        metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
        columns_to_show.extend(metric_cols)
        
        # Add all params
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        columns_to_show.extend(param_cols)
        
        # Add tags if exist
        if 'tags.run_name' in runs.columns:
            columns_to_show.insert(1, 'tags.run_name')
        
        comparison = runs[columns_to_show].head(top_n)
        
        return comparison
    
    def get_best_run(self, metric_name: str = "rmse", 
                     ascending: bool = True) -> Dict:
        """
        Get the best run based on a metric
        
        Args:
            metric_name: Metric to optimize
            ascending: True if lower is better (e.g., RMSE), False if higher is better
        
        Returns:
            Dictionary with best run info
        """
        order = "ASC" if ascending else "DESC"
        
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {order}"]
        )
        
        if len(runs) == 0:
            return {}
        
        best_run = runs.iloc[0]
        
        return {
            'run_id': best_run['run_id'],
            'run_name': best_run.get('tags.run_name', 'unknown'),
            'metric_value': best_run[f'metrics.{metric_name}'],
            'params': {k.replace('params.', ''): v 
                      for k, v in best_run.items() 
                      if k.startswith('params.')},
            'all_metrics': {k.replace('metrics.', ''): v 
                           for k, v in best_run.items() 
                           if k.startswith('metrics.')}
        }


def track_model_training(tracker: ExperimentTracker,
                        run_name: str,
                        model,
                        train_func,
                        train_data,
                        val_data,
                        test_data,
                        algorithm_name: str,
                        hyperparams: Dict):
    """
    Complete training workflow with tracking
    
    Args:
        tracker: ExperimentTracker instance
        run_name: Name for this run
        model: Model instance (SVDRecommender or ALSRecommender)
        train_func: Function to train model
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame
        algorithm_name: "SVD" or "ALS"
        hyperparams: Dictionary of hyperparameters
    
    Returns:
        Dictionary of results
    """
    from src.evaluation.evaluation_metrics import RecommendationEvaluator
    
    with tracker.start_run(run_name=run_name, tags={'algorithm': algorithm_name}):
        # Log hyperparameters
        tracker.log_params(hyperparams)
        tracker.log_params({
            'algorithm': algorithm_name,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        })
        
        # Train model
        print(f"\n{'='*60}")
        print(f"Training {algorithm_name} - Run: {run_name}")
        print(f"{'='*60}")
        
        train_stats = train_func(train_data, verbose=True)
        
        # Log training stats
        tracker.log_metrics({
            'training_time_seconds': train_stats.get('training_time', 0)
        })
        
        # Evaluate on validation set
        print("\n📊 Evaluating on validation set...")
        evaluator = RecommendationEvaluator()
        
        # Sample for faster evaluation
        val_sample = val_data.sample(min(1000, len(val_data)))
        
        # Accuracy metrics
        true_ratings = []
        pred_ratings = []
        
        for _, row in val_sample.iterrows():
            try:
                if algorithm_name == "ALS":
                    pred, _ = model.predict(row['userId'], row['movieId'])
                else:
                    pred, _ = model.predict(row['userId'], row['movieId'])
                true_ratings.append(row['rating'])
                pred_ratings.append(pred)
            except:
                continue
        
        if len(true_ratings) > 0:
            val_rmse = evaluator.rmse(true_ratings, pred_ratings)
            val_mae = evaluator.mae(true_ratings, pred_ratings)
            
            tracker.log_metrics({
                'val_rmse': val_rmse,
                'val_mae': val_mae
            })
            
            print(f"✅ Validation RMSE: {val_rmse:.4f}")
            print(f"✅ Validation MAE: {val_mae:.4f}")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_sample = test_data.sample(min(1000, len(test_data)))
        
        true_ratings = []
        pred_ratings = []
        
        for _, row in test_sample.iterrows():
            try:
                if algorithm_name == "ALS":
                    pred, _ = model.predict(row['userId'], row['movieId'])
                else:
                    pred, _ = model.predict(row['userId'], row['movieId'])
                true_ratings.append(row['rating'])
                pred_ratings.append(pred)
            except:
                continue
        
        results = {}
        if len(true_ratings) > 0:
            test_rmse = evaluator.rmse(true_ratings, pred_ratings)
            test_mae = evaluator.mae(true_ratings, pred_ratings)
            
            tracker.log_metrics({
                'test_rmse': test_rmse,
                'test_mae': test_mae
            })
            
            print(f"✅ Test RMSE: {test_rmse:.4f}")
            print(f"✅ Test MAE: {test_mae:.4f}")
            
            results = {
                'algorithm': algorithm_name,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'training_time': train_stats.get('training_time', 0)
            }
        
        # Save model
        model_path = f"models/{algorithm_name.lower()}_model.pkl"
        model.save_model(model_path)
        
        # Log model artifact
        tracker.log_dict(results, f"{algorithm_name}_results.json")
        
        print(f"\n✅ Run '{run_name}' complete and logged to MLflow")
        
        return results


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker("recommendation_system")
    
    print("\nTo view experiments, run:")
    print("  mlflow ui")
    print("Then open: http://localhost:5000")


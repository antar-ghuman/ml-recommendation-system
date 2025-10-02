"""
Evaluation Metrics for Recommendation Systems
Includes accuracy, ranking, diversity, and A/B testing metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import yaml


class RecommendationEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems
    Covers both offline metrics and A/B testing analysis
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.metrics_config = self.config.get('metrics', [])
    
    # ========== ACCURACY METRICS ==========
    
    def rmse(self, true_ratings: List[float], predicted_ratings: List[float]) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    
    def mae(self, true_ratings: List[float], predicted_ratings: List[float]) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(true_ratings, predicted_ratings)
    
    # ========== RANKING METRICS ==========
    
    def precision_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Precision@K: What fraction of recommendations are relevant?
        
        Args:
            recommended: List of recommended item IDs (ordered by rank)
            relevant: List of relevant item IDs (ground truth)
            k: Number of recommendations to consider
        """
        if k == 0:
            return 0.0
        
        recommended_at_k = recommended[:k]
        relevant_set = set(relevant)
        
        num_relevant = len([item for item in recommended_at_k if item in relevant_set])
        return num_relevant / k
    
    def recall_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Recall@K: What fraction of relevant items are recommended?
        
        Args:
            recommended: List of recommended item IDs (ordered by rank)
            relevant: List of relevant item IDs (ground truth)
            k: Number of recommendations to consider
        """
        if len(relevant) == 0:
            return 0.0
        
        recommended_at_k = recommended[:k]
        relevant_set = set(relevant)
        
        num_relevant = len([item for item in recommended_at_k if item in relevant_set])
        return num_relevant / len(relevant)
    
    def average_precision(self, recommended: List[int], relevant: List[int]) -> float:
        """
        Average Precision: Precision averaged over all positions where relevant items appear
        """
        if len(relevant) == 0:
            return 0.0
        
        relevant_set = set(relevant)
        score = 0.0
        num_relevant = 0
        
        for i, item in enumerate(recommended, 1):
            if item in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / i
                score += precision_at_i
        
        return score / len(relevant) if len(relevant) > 0 else 0.0
    
    def ndcg_at_k(self, recommended: List[int], relevant: List[int], 
                  relevance_scores: Dict[int, float] = None, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        Accounts for position of relevant items
        
        Args:
            recommended: Ordered list of recommended items
            relevant: List of relevant items
            relevance_scores: Dict mapping item_id to relevance score (optional)
            k: Number of recommendations to consider
        """
        def dcg(items, scores, k):
            """Calculate DCG"""
            dcg_score = 0.0
            for i, item in enumerate(items[:k], 1):
                relevance = scores.get(item, 0.0)
                dcg_score += relevance / np.log2(i + 1)
            return dcg_score
        
        # If no scores provided, use binary relevance (1 if relevant, 0 otherwise)
        if relevance_scores is None:
            relevant_set = set(relevant)
            relevance_scores = {item: 1.0 for item in relevant_set}
        
        # Calculate DCG for recommendations
        actual_dcg = dcg(recommended, relevance_scores, k)
        
        # Calculate ideal DCG (best possible ordering)
        ideal_order = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        ideal_items = [item for item, _ in ideal_order]
        ideal_dcg = dcg(ideal_items, relevance_scores, k)
        
        # Normalize
        if ideal_dcg == 0:
            return 0.0
        
        return actual_dcg / ideal_dcg
    
    # ========== DIVERSITY & COVERAGE METRICS ==========
    
    def catalog_coverage(self, all_recommendations: List[List[int]], 
                        total_items: int) -> float:
        """
        What percentage of the catalog appears in recommendations?
        
        Args:
            all_recommendations: List of recommendation lists for all users
            total_items: Total number of items in catalog
        """
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update(recs)
        
        return len(unique_recommended) / total_items
    
    def diversity(self, recommendations: List[int], 
                 item_similarity_matrix: np.ndarray = None,
                 item_features: Dict[int, List] = None) -> float:
        """
        Diversity: How different are items from each other?
        
        Can use either:
        - Precomputed similarity matrix
        - Item features (will compute pairwise similarity)
        """
        if len(recommendations) <= 1:
            return 1.0
        
        # If we have similarity matrix, use it
        if item_similarity_matrix is not None:
            # Average pairwise dissimilarity
            total_dissimilarity = 0.0
            count = 0
            for i in range(len(recommendations)):
                for j in range(i + 1, len(recommendations)):
                    item_i, item_j = recommendations[i], recommendations[j]
                    similarity = item_similarity_matrix[item_i, item_j]
                    total_dissimilarity += (1 - similarity)
                    count += 1
            return total_dissimilarity / count if count > 0 else 0.0
        
        # If we have features, compute diversity using feature overlap
        elif item_features is not None:
            total_diversity = 0.0
            count = 0
            for i in range(len(recommendations)):
                for j in range(i + 1, len(recommendations)):
                    item_i = set(item_features.get(recommendations[i], []))
                    item_j = set(item_features.get(recommendations[j], []))
                    
                    # Jaccard distance
                    if len(item_i | item_j) > 0:
                        similarity = len(item_i & item_j) / len(item_i | item_j)
                        total_diversity += (1 - similarity)
                    count += 1
            return total_diversity / count if count > 0 else 0.0
        
        else:
            raise ValueError("Must provide either similarity_matrix or item_features")
    
    def novelty(self, recommendations: List[int], 
                item_popularity: Dict[int, int],
                total_interactions: int) -> float:
        """
        Novelty: How surprising/non-obvious are the recommendations?
        Based on item popularity
        
        Args:
            recommendations: List of recommended items
            item_popularity: Dict mapping item_id to number of interactions
            total_interactions: Total number of interactions in system
        """
        novelty_scores = []
        for item in recommendations:
            popularity = item_popularity.get(item, 0)
            probability = popularity / total_interactions if total_interactions > 0 else 0
            # Self-information: -log2(p)
            if probability > 0:
                novelty_scores.append(-np.log2(probability))
            else:
                novelty_scores.append(0)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    # ========== BATCH EVALUATION ==========
    
    def evaluate_model(self, model, test_df: pd.DataFrame, 
                      k_values: List[int] = [5, 10, 20],
                      calculate_diversity: bool = False) -> Dict:
        """
        Comprehensive evaluation of a recommendation model
        
        Args:
            model: Trained recommendation model with predict() method
            test_df: Test dataset
            k_values: List of K values for ranking metrics
            calculate_diversity: Whether to calculate diversity metrics
        
        Returns:
            Dictionary of metric results
        """
        print("Evaluating model...")
        
        results = {}
        
        # 1. Accuracy metrics (rating prediction)
        true_ratings = []
        pred_ratings = []
        
        for _, row in test_df.iterrows():
            try:
                pred, _ = model.predict(row['userId'], row['movieId'])
                true_ratings.append(row['rating'])
                pred_ratings.append(pred)
            except:
                continue
        
        if len(true_ratings) > 0:
            results['rmse'] = self.rmse(true_ratings, pred_ratings)
            results['mae'] = self.mae(true_ratings, pred_ratings)
            print(f"✅ RMSE: {results['rmse']:.4f}")
            print(f"✅ MAE: {results['mae']:.4f}")
        
        # 2. Ranking metrics
        # Get user-item interactions for evaluation
        user_interactions = test_df.groupby('userId')['movieId'].apply(list).to_dict()
        
        # Sample users for ranking evaluation (can be slow for all users)
        sample_users = list(user_interactions.keys())[:1000]  # Evaluate on 1000 users
        
        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []
            
            for user_id in sample_users:
                try:
                    # Get recommendations
                    recs = model.recommend_for_user(user_id, n=k)
                    recommended = recs['movieId'].tolist()
                    
                    # Get relevant items (items user actually interacted with in test)
                    relevant = user_interactions.get(user_id, [])
                    
                    if len(relevant) > 0:
                        precisions.append(self.precision_at_k(recommended, relevant, k))
                        recalls.append(self.recall_at_k(recommended, relevant, k))
                        ndcgs.append(self.ndcg_at_k(recommended, relevant, k=k))
                except:
                    continue
            
            if len(precisions) > 0:
                results[f'precision@{k}'] = np.mean(precisions)
                results[f'recall@{k}'] = np.mean(recalls)
                results[f'ndcg@{k}'] = np.mean(ndcgs)
                print(f"✅ Precision@{k}: {results[f'precision@{k}']:.4f}")
                print(f"✅ Recall@{k}: {results[f'recall@{k}']:.4f}")
                print(f"✅ NDCG@{k}: {results[f'ndcg@{k}']:.4f}")
        
        return results
    
    # ========== A/B TESTING FRAMEWORK ==========
    
    def assign_to_group(self, user_id: int, salt: str = "experiment_001") -> str:
        """
        Consistently assign users to A/B test groups
        
        Args:
            user_id: User ID
            salt: Experiment identifier (different salts = different splits)
        
        Returns:
            'control' or 'treatment'
        """
        # Use hash for consistent assignment
        hash_value = hash(f"{user_id}_{salt}")
        return 'treatment' if hash_value % 2 == 0 else 'control'
    
    def ab_test_analysis(self, control_metrics: List[float], 
                        treatment_metrics: List[float],
                        metric_name: str = "metric",
                        confidence_level: float = 0.95) -> Dict:
        """
        Statistical analysis of A/B test results
        
        Args:
            control_metrics: List of metric values from control group
            treatment_metrics: List of metric values from treatment group
            metric_name: Name of the metric being tested
            confidence_level: Confidence level for significance test
        
        Returns:
            Dictionary with test results
        """
        control_mean = np.mean(control_metrics)
        treatment_mean = np.mean(treatment_metrics)
        
        # Calculate relative lift
        lift = (treatment_mean - control_mean) / control_mean * 100
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment_metrics, control_metrics)
        
        # Check statistical significance
        is_significant = p_value < (1 - confidence_level)
        
        # Calculate confidence interval for lift
        pooled_std = np.sqrt(
            (np.var(control_metrics) / len(control_metrics)) + 
            (np.var(treatment_metrics) / len(treatment_metrics))
        )
        critical_value = stats.t.ppf((1 + confidence_level) / 2, 
                                     len(control_metrics) + len(treatment_metrics) - 2)
        margin_of_error = critical_value * pooled_std
        
        results = {
            'metric_name': metric_name,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'absolute_difference': treatment_mean - control_mean,
            'relative_lift_pct': lift,
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_level': confidence_level,
            'confidence_interval': (
                treatment_mean - margin_of_error,
                treatment_mean + margin_of_error
            ),
            'sample_size_control': len(control_metrics),
            'sample_size_treatment': len(treatment_metrics),
            't_statistic': t_stat
        }
        
        return results
    
    def print_ab_results(self, results: Dict) -> None:
        """Pretty print A/B test results"""
        print("\n" + "="*60)
        print(f"A/B TEST RESULTS: {results['metric_name']}")
        print("="*60)
        print(f"Control Group:")
        print(f"  Mean: {results['control_mean']:.4f}")
        print(f"  Sample size: {results['sample_size_control']:,}")
        print(f"\nTreatment Group:")
        print(f"  Mean: {results['treatment_mean']:.4f}")
        print(f"  Sample size: {results['sample_size_treatment']:,}")
        print(f"\nResults:")
        print(f"  Absolute difference: {results['absolute_difference']:.4f}")
        print(f"  Relative lift: {results['relative_lift_pct']:.2f}%")
        print(f"  P-value: {results['p_value']:.4f}")
        print(f"  Statistical significance ({results['confidence_level']*100:.0f}% confidence): {'YES ✅' if results['is_significant'] else 'NO ❌'}")
        
        if results['is_significant']:
            if results['relative_lift_pct'] > 0:
                print(f"\n🎉 Treatment is SIGNIFICANTLY BETTER than control!")
            else:
                print(f"\n⚠️  Treatment is SIGNIFICANTLY WORSE than control!")
        else:
            print(f"\n➡️  No significant difference detected. Need more data or bigger effect.")
        print("="*60)
    
    def sample_size_calculator(self, baseline_rate: float, 
                              minimum_detectable_effect: float,
                              power: float = 0.8,
                              alpha: float = 0.05) -> int:
        """
        Calculate required sample size for A/B test
        
        Args:
            baseline_rate: Current metric value (e.g., 0.15 for 15% CTR)
            minimum_detectable_effect: Minimum relative change to detect (e.g., 0.05 for 5%)
            power: Statistical power (typically 0.8)
            alpha: Significance level (typically 0.05)
        
        Returns:
            Required sample size per group
        """
        # Using approximation for proportions
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        n = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
        
        return int(np.ceil(n))


if __name__ == "__main__":
    # Example usage
    evaluator = RecommendationEvaluator()
    
    # Example: Calculate required sample size
    print("Sample size calculator:")
    sample_size = evaluator.sample_size_calculator(
        baseline_rate=0.10,  # 10% click-through rate
        minimum_detectable_effect=0.10,  # Want to detect 10% improvement
        power=0.8,
        alpha=0.05
    )
    print(f"Required sample size per group: {sample_size:,}")
    
    # Example A/B test
    print("\nExample A/B test:")
    control = np.random.normal(0.10, 0.02, 1000)
    treatment = np.random.normal(0.11, 0.02, 1000)  # 10% improvement
    
    results = evaluator.ab_test_analysis(control, treatment, "Click-Through Rate")
    evaluator.print_ab_results(results)
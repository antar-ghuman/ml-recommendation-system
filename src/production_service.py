"""
Production Features for Recommendation System
Demonstrates production ML engineering skills
"""

import time
from functools import lru_cache
from typing import List, Dict, Optional
import pandas as pd
import pickle
from pathlib import Path
import hashlib
import json
from datetime import datetime, timedelta


class ProductionRecommender:
    """
    Production-ready recommendation service with:
    - Caching
    - Fallback strategies
    - Monitoring
    - A/B testing support
    - Performance optimization
    """
    
    def __init__(self, model_path: str = "models/svd_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.cache = {}
        self.metrics = {
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'avg_latency_ms': 0,
            'cold_start_fallbacks': 0
        }
        self.load_model()
    
    def load_model(self):
        """Load model with error handling"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Import the appropriate class
            from src.algorithms.svd_algorithm import SVDRecommender
            self.model = SVDRecommender()
            
            # Load model state
            self.model.model = model_data['model']
            self.model.config = model_data['config']
            self.model.train_stats = model_data['train_stats']
            self.model.user_map = model_data['user_map']
            self.model.movie_map = model_data['movie_map']
            
            print(f"✅ Model loaded from {self.model_path}")
            print(f"   Users: {len(self.model.user_map):,}")
            print(f"   Movies: {len(self.model.movie_map):,}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    @lru_cache(maxsize=10000)
    def get_recommendations_cached(
        self, 
        user_id: int, 
        n: int = 10,
        exclude_tuple: tuple = ()
    ) -> str:
        """
        Cached recommendations (LRU cache for hot users)
        
        Note: exclude_seen must be tuple for caching to work
        Returns JSON string for caching
        """
        exclude_list = list(exclude_tuple) if exclude_tuple else None
        
        try:
            recs = self.model.recommend_for_user(
                user_id, 
                n=n, 
                exclude_seen=exclude_list
            )
            return recs.to_json()
        except Exception as e:
            return json.dumps({'error': str(e)})
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: List[int] = None,
        use_cache: bool = True,
        experiment_group: str = 'control'
    ) -> Dict:
        """
        Production recommendation endpoint
        
        Features:
        - Caching
        - Monitoring
        - Fallback strategies
        - A/B testing support
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_seen: Movies to exclude
            use_cache: Whether to use cache
            experiment_group: A/B test group ('control' or 'treatment')
        
        Returns:
            Dict with recommendations and metadata
        """
        start_time = time.time()
        self.metrics['requests'] += 1
        
        try:
            # Check cache
            cache_key = f"{user_id}_{n}_{experiment_group}"
            
            if use_cache and cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    'recommendations': self.cache[cache_key],
                    'user_id': user_id,
                    'n': n,
                    'experiment_group': experiment_group,
                    'cache_hit': True,
                    'latency_ms': latency_ms,
                    'timestamp': datetime.now().isoformat()
                }
            
            self.metrics['cache_misses'] += 1
            
            # Check if user exists
            if user_id not in self.model.user_map:
                # Cold start: fallback to popular items
                self.metrics['cold_start_fallbacks'] += 1
                recs = self._get_popular_items(n)
                strategy = 'popular_fallback'
            else:
                # Normal recommendation
                recs = self.model.recommend_for_user(
                    user_id, 
                    n=n, 
                    exclude_seen=exclude_seen
                )
                strategy = 'collaborative_filtering'
            
            # Convert to dict
            recs_list = recs.to_dict('records')
            
            # Cache results
            if use_cache:
                self.cache[cache_key] = recs_list
            
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)
            
            return {
                'recommendations': recs_list,
                'user_id': user_id,
                'n': n,
                'experiment_group': experiment_group,
                'strategy': strategy,
                'cache_hit': False,
                'latency_ms': latency_ms,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.metrics['errors'] += 1
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'error': str(e),
                'user_id': user_id,
                'latency_ms': latency_ms,
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_popular_items(self, n: int = 10) -> pd.DataFrame:
        """Fallback: return popular items"""
        # Placeholder - in production, query from database
        # For now, return random movies from our catalog
        import random
        movie_ids = list(self.model.movie_map.keys())
        popular = random.sample(movie_ids, min(n, len(movie_ids)))
        
        return pd.DataFrame([
            {'movieId': mid, 'predicted_rating': 4.0, 'rank': i+1}
            for i, mid in enumerate(popular)
        ])
    
    def _update_latency(self, latency_ms: float):
        """Update rolling average latency"""
        alpha = 0.1  # Exponential moving average factor
        if self.metrics['avg_latency_ms'] == 0:
            self.metrics['avg_latency_ms'] = latency_ms
        else:
            self.metrics['avg_latency_ms'] = (
                alpha * latency_ms + 
                (1 - alpha) * self.metrics['avg_latency_ms']
            )
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        total_requests = self.metrics['requests']
        
        if total_requests == 0:
            return self.metrics
        
        cache_hit_rate = self.metrics['cache_hits'] / total_requests
        error_rate = self.metrics['errors'] / total_requests
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'error_rate': error_rate
        }
    
    def health_check(self) -> Dict:
        """Health check endpoint"""
        metrics = self.get_metrics()
        
        is_healthy = (
            self.model is not None and
            metrics['error_rate'] < 0.05 and  # Less than 5% errors
            metrics['avg_latency_ms'] < 100   # Less than 100ms average
        )
        
        return {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'model_loaded': self.model is not None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear recommendation cache"""
        self.cache.clear()
        print("✅ Cache cleared")
    
    def ab_test_assignment(self, user_id: int, experiment_name: str = "default") -> str:
        """
        Consistent A/B test assignment using hashing
        
        Args:
            user_id: User ID
            experiment_name: Experiment name (salt)
        
        Returns:
            'control' or 'treatment'
        """
        # Hash user_id + experiment name for consistency
        hash_input = f"{user_id}_{experiment_name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        return 'treatment' if hash_value % 2 == 0 else 'control'


class ModelMonitor:
    """
    Monitor model performance in production
    Tracks metrics and alerts on degradation
    """
    
    def __init__(self, alert_threshold: float = 0.1):
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.alert_threshold = alert_threshold  # 10% degradation triggers alert
        self.alerts = []
    
    def set_baseline(self, metrics: Dict):
        """Set baseline metrics from validation"""
        self.baseline_metrics = metrics
        print(f"✅ Baseline set: RMSE={metrics.get('rmse', 0):.4f}")
    
    def log_prediction(self, user_id: int, movie_id: int, 
                      predicted: float, actual: float = None):
        """Log a prediction for monitoring"""
        if actual is not None:
            error = abs(predicted - actual)
            
            if 'errors' not in self.current_metrics:
                self.current_metrics['errors'] = []
            
            self.current_metrics['errors'].append(error)
            
            # Check for degradation every 100 predictions
            if len(self.current_metrics['errors']) >= 100:
                self._check_degradation()
    
    def _check_degradation(self):
        """Check if model performance has degraded"""
        if 'errors' not in self.current_metrics:
            return
        
        current_mae = sum(self.current_metrics['errors']) / len(self.current_metrics['errors'])
        baseline_mae = self.baseline_metrics.get('mae', current_mae)
        
        degradation = (current_mae - baseline_mae) / baseline_mae
        
        if degradation > self.alert_threshold:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'metric': 'MAE',
                'baseline': baseline_mae,
                'current': current_mae,
                'degradation_pct': degradation * 100,
                'message': f"⚠️  Model degradation detected! MAE increased by {degradation*100:.1f}%"
            }
            self.alerts.append(alert)
            print(alert['message'])
        
        # Reset for next batch
        self.current_metrics['errors'] = []
    
    def get_alerts(self) -> List[Dict]:
        """Get all alerts"""
        return self.alerts


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("PRODUCTION RECOMMENDATION SERVICE - DEMO")
    print("="*70 + "\n")
    
    # Initialize service
    service = ProductionRecommender()
    
    # Test recommendations
    print("\n1. Getting recommendations for user 131073...")
    result = service.recommend(user_id=131073, n=5)
    
    print(f"   Strategy: {result['strategy']}")
    print(f"   Latency: {result['latency_ms']:.2f}ms")
    print(f"   Cache hit: {result['cache_hit']}")
    print(f"   Recommendations: {len(result['recommendations'])}")
    
    # Test caching
    print("\n2. Same request again (should hit cache)...")
    result2 = service.recommend(user_id=131073, n=5)
    print(f"   Cache hit: {result2['cache_hit']}")
    print(f"   Latency: {result2['latency_ms']:.2f}ms (faster!)")
    
    # Test cold start
    print("\n3. Cold start user (not in training data)...")
    result3 = service.recommend(user_id=999999999, n=5)
    print(f"   Strategy: {result3['strategy']}")
    
    # A/B test assignment
    print("\n4. A/B test assignment...")
    for user_id in [1, 2, 3, 4, 5]:
        group = service.ab_test_assignment(user_id)
        print(f"   User {user_id}: {group}")
    
    # Metrics
    print("\n5. Service metrics...")
    metrics = service.get_metrics()
    print(f"   Total requests: {metrics['requests']}")
    print(f"   Cache hit rate: {metrics['cache_hit_rate']*100:.1f}%")
    print(f"   Avg latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"   Error rate: {metrics['error_rate']*100:.1f}%")
    
    # Health check
    print("\n6. Health check...")
    health = service.health_check()
    print(f"   Status: {health['status']}")
    
    print("\n" + "="*70)
    print("✅ Production features demo complete!")
    print("="*70)
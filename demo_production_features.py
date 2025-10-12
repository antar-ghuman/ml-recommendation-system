"""
Demo: Production Features
Run from project root
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.production_service import ProductionRecommender, ModelMonitor

print("="*70)
print("PRODUCTION RECOMMENDATION SERVICE - DEMO")
print("="*70 + "\n")

# Initialize service
service = ProductionRecommender(model_path="models/svd_model.pkl")

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
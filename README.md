# Movie Recommendation System

A recommendation system implementing collaborative filtering with SVD, comprehensive experiment tracking, and A/B testing framework.

## Results

* **Test RMSE:** 0.8400 on MovieLens 25M dataset
* **Training Time:** 12 seconds on 2.4M ratings
* **Dataset:** MovieLens 25M (25 million ratings, 162k users, 59k movies)

## What This Project Demonstrates

* SVD collaborative filtering implementation
* Hyperparameter tuning across 9 configurations
* A/B testing with statistical significance analysis
* MLflow experiment tracking
* Production-ready code structure with monitoring and health checks
* Proper train/validation/test methodology

## Algorithms

**SVD (Matrix Factorization)**
* Optimized for explicit ratings (1-5 stars)
* Test RMSE: 0.8400
* Training: 12 seconds on 2.4M ratings

**ALS (Alternating Least Squares)**
* Designed for implicit feedback (clicks, views)
* Test RMSE: 3.4885 on explicit ratings
* Demonstrates understanding of algorithm selection for different data types

## Key Features

**Experiment Tracking:**
* MLflow integration for all model runs
* Comprehensive logging of metrics and hyperparameters
* Visual comparison across configurations

**A/B Testing Framework:**
* Statistical significance testing using t-tests
* Consistent user assignment via hashing
* Sample size calculator

**Evaluation:**
* Accuracy metrics: RMSE, MAE
* Ranking metrics: Precision@K, NDCG@K
* Proper data splits (70/10/20) to prevent leakage

## Hyperparameter Tuning Results

Tested 9 configurations varying:
* Factors: 50-100
* Epochs: 10-15
* Learning rate: 0.005-0.007
* Regularization: 0.02-0.04

**Best configuration:** 100 factors, 15 epochs, lr=0.007, reg=0.02

## Tech Stack

* Python 3.12
* scikit-surprise (SVD)
* implicit (ALS)
* MLflow (experiment tracking)
* pandas, numpy (data processing)

## What I Learned

1. **Algorithm selection:** SVD excels at explicit ratings; ALS designed for implicit feedback
2. **Hyperparameter impact:** Increasing factors from 50→100 improved performance
3. **Evaluation methodology:** Proper train/val/test splits prevent data leakage
4. **Experiment tracking:** MLflow enables systematic comparison of model variants
5. **Statistical testing:** A/B testing framework ensures changes are significant, not random

## Future Improvements

* FastAPI REST API for model serving
* Docker containerization
* Additional algorithms (Neural Collaborative Filtering, content-based filtering)
* Caching layer for frequently requested recommendations
* Real-time model updates

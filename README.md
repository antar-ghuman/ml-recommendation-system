# Multi-Algorithm Recommendation System

A production-ready recommendation system with A/B testing framework, built for demonstrating ML engineering skills.

## Features

- **Multiple Algorithms**: SVD, ALS, Neural Collaborative Filtering, Content-Based
- **A/B Testing Framework**: Statistical significance testing and experiment tracking
- **Production-Ready**: FastAPI service, proper evaluation metrics, modular code
- **Comprehensive Evaluation**: RMSE, Precision@K, NDCG, Diversity metrics

## Project Structure

```
ml-recommendation-system/
├── data/
│   ├── raw/              # Original MovieLens dataset
│   └── processed/        # Cleaned, split data
├── models/               # Saved model files
├── src/
│   ├── algorithms/       # Recommendation algorithms
│   ├── evaluation/       # Metrics and A/B testing
│   └── api/             # FastAPI service
├── config/              # YAML configuration files
├── scripts/             # Utility scripts
└── experiments/         # A/B test results
```

## Quick Start

### 1. Download Data
```bash
python scripts/download_data.py
```

### 2. Run Full Pipeline
```bash
python run_pipeline.py
```

### 3. Start API Server (coming soon)
```bash
uvicorn src.api.main:app --reload
```

## Algorithms Implemented

### 1. SVD (Singular Value Decomposition)
- Classic matrix factorization
- Netflix Prize winning approach
- Fast training and inference

### 2. ALS (Alternating Least Squares) [Coming Soon]
- Implicit feedback support
- Parallel training
- Good for large-scale systems

### 3. Neural Collaborative Filtering [Coming Soon]
- Deep learning approach
- Learns non-linear patterns
- State-of-the-art performance

### 4. Content-Based Filtering [Coming Soon]
- Uses item features
- Good for cold-start items
- Interpretable recommendations

## Evaluation Metrics

### Accuracy
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### Ranking
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)

### Diversity
- Catalog Coverage
- Intra-list Diversity
- Novelty

## A/B Testing

The framework includes:
- User group assignment (consistent hashing)
- Statistical significance testing (t-tests)
- Sample size calculator
- Experiment tracking with MLflow

## Tech Stack

- **ML Libraries**: scikit-learn, Surprise, PyTorch, TensorFlow
- **API**: FastAPI, Uvicorn
- **Experiment Tracking**: MLflow
- **Data Processing**: Pandas, NumPy
- **Testing**: Pytest

## Development Roadmap

- [x] Data preprocessing pipeline
- [x] SVD algorithm implementation
- [x] Evaluation framework
- [ ] Additional algorithms (ALS, NCF, Content-Based)
- [ ] FastAPI service
- [ ] MLflow integration
- [ ] Docker containerization
- [ ] CI/CD pipeline

## Dataset

Using MovieLens 25M dataset:
- 25 million ratings
- 62,000 movies
- 162,000 users

## Author

Built as a portfolio project demonstrating production ML system design.

## License

MIT

# Kaggle Playground Series S5E8: Bank Register Prediction

This repository contains a machine-learning workflow for the Kaggle competition **[Playground Series - Season 5, Episode 8](https://www.kaggle.com/competitions/playground-series-s5e8)**. The goal is to build predictive models for the competition target and generate high-quality submissions.

## Project Overview

The project is structured to support an end-to-end experimentation loop:

1. Load and inspect competition data.
2. Build reproducible preprocessing and feature engineering steps.
3. Train and compare multiple models.
4. Tune model hyperparameters.
5. Evaluate with the competition metric.
6. Export submission files for Kaggle.

## Tech Stack

Core dependencies are listed in `requirement.txt`:

- `pandas`, `numpy` for data processing
- `scikit-learn` for baseline models and pipelines
- `xgboost`, `lightgbm` for gradient boosting models
- `optuna` for hyperparameter optimization
- `matplotlib`, `seaborn` for exploratory analysis and visualization

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

## Suggested Repository Workflow

- `data/`: local copies of competition datasets (do not commit raw data)
- `notebooks/`: exploratory analysis and prototype experiments
- `src/`: reusable training, inference, and utility code
- `outputs/`: model artifacts, plots, and submission CSVs

## Competition Reference

- Kaggle competition page: https://www.kaggle.com/competitions/playground-series-s5e8

## Notes

- Keep experiments reproducible by fixing random seeds and tracking parameters.
- Validate locally before submission to avoid leaderboard overfitting.
- Use Git commits to document changes in features, models, and results.
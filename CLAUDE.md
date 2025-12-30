# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TabularAML is an automated feature engineering library for tabular machine learning. It uses a genetic algorithm to discover and create new features through mathematical transformations of existing features. The project is being used for the DrivenData "Poverty Prediction Challenge" (World Bank competition).

## Build and Development Commands

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install flask flask-socketio xgboost pandas numpy tqdm cloudpickle shap lightgbm

# Run the UI server
python tabularaml/ui/feature_generator_server.py
# Serves at http://localhost:5000

# Run tests (if available)
pytest -q
pytest -k <pattern>  # Run specific tests

# Lint/format
ruff check .
black .
```

## Architecture

### Core Feature Generation Pipeline

The system uses a genetic algorithm approach:

1. **`FeatureGenerator`** (`tabularaml/generate/features.py`): Main orchestrator
   - `search()`: Runs the evolutionary search over generations
   - Uses adaptive stagnation handling with `ImprovedAdaptiveController`
   - Supports modes: `lite`, `medium`, `best`, `extreme` (defined in `configs/feature_gen.py`)
   - Maintains state tracking for best features, early stopping, and restarts

2. **`Feature` and `Interaction`** classes: Represent individual features and their transformations
   - Features have dtype (`num`/`cat`), weight, depth, and pipeline requirements
   - Interactions combine features via operations (unary or binary)

3. **Operations** (`tabularaml/generate/ops.py`):
   - `OPS` dict defines available operations per dtype
   - Numeric unary: neg, abs, square, sqrt, log, exp, sin, cos, sigmoid, etc.
   - Numeric binary: add, sub, mul, div, ratio, geometric_mean, etc.
   - Categorical unary: target, freq, count (require pipeline for CV)
   - Categorical binary: concat

### Preprocessing Pipeline

`PipelineWrapper` (`tabularaml/preprocessing/pipeline.py`):
- Wraps sklearn's ColumnTransformer with imputer, scaler, encoder
- Auto-detects column dtypes and handles conversions
- `CategoricalEncoder` handles target/count/frequency encoding

### Evaluation System

- **`cross_val_score`** (`tabularaml/eval/cv.py`): Custom CV implementation supporting pipelines and fold weights
- **`Scorer`** (`tabularaml/eval/scorers.py`): Unified scorer interface for XGBoost/LightGBM/CatBoost
- **`FeatureImportanceAnalyzer`** (`tabularaml/inspect/importance.py`): Multi-method importance (tree, SHAP, permutation, correlation)

### Preset Configurations

Located in `tabularaml/configs/feature_gen.py`:
- `lite`: Fast exploration (5 min budget, 12 generations)
- `medium`: Balanced (15 min budget, 25 generations)
- `best`: Deep search (2 hour budget, 45 generations)
- `extreme`: Maximum thoroughness (4 hour budget, 80 generations)

## Key Patterns

### Feature Generator Usage

```python
from tabularaml.generate.features import FeatureGenerator

# Using preset mode
gen = FeatureGenerator(mode="medium", task="regression")
X_new, pipeline, generation, interactions = gen.search(X, y)

# Fit/transform for inference
gen.fit(X_train, y_train)
X_test_transformed = gen.transform(X_test)

# Save/load
gen.save("model.pkl")
gen = FeatureGenerator.load("model.pkl")
```

### Scorer Usage

```python
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS

scorer = PREDEFINED_REG_SCORERS["rmse"]  # or mae, mse, r2
scorer = PREDEFINED_CLS_SCORERS["binary_crossentropy"]  # or accuracy, f1, etc.
```

## Competition Context (Poverty Prediction)

- **Task**: Predict household consumption and poverty rates from survey data
- **Metric**: 90% weighted MAPE on poverty rates + 10% MAPE on consumption
- **Data files** in `data/`:
  - `train_hh_features.csv`, `train_hh_gt.csv`, `train_rates_gt.csv`
  - `test_hh_features.csv`
- Surveys have IDs 100000-600000; train=100000,200000,300000; test=400000,500000,600000
- Submission format: zip with `predicted_household_consumption.csv` and `predicted_poverty_distribution.csv`

## Output Directories

- `cache/`: Artifacts, logs, checkpoints
- `logs/`: Feature generation search logs
- Model persistence uses `cloudpickle`

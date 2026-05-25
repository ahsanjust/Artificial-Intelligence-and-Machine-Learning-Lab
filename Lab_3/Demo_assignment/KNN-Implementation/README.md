# KNN Implementation (Breast Cancer Classification)

This repository contains a simple end-to-end K-Nearest Neighbors (KNN) classification workflow on the Breast Cancer Wisconsin (Diagnostic) dataset using Python and scikit-learn.

The notebook covers:
- Loading the dataset
- Label encoding the target (`diagnosis`)
- Train/validation/test split
- Feature scaling with `StandardScaler`
- Selecting $k$ via a validation error curve
- Hyperparameter tuning with `GridSearchCV`
- Evaluation with confusion matrix, ROC curve/AUC, and common classification metrics
- 2D PCA visualization with a KNN decision surface

## Project Structure

- `210116_KNN.ipynb` — main notebook (implementation + plots)
- `data.csv` — dataset CSV used by the notebook

## Dataset

The dataset is the Breast Cancer Wisconsin (Diagnostic) dataset.

- Target column: `diagnosis`
	- `B` = benign
	- `M` = malignant

In the notebook, the target is converted to numeric labels using `LabelEncoder`. With scikit-learn’s default behavior this typically maps `B → 0` and `M → 1`.

Note: `data.csv` includes a trailing empty column which pandas reads as `Unnamed: 32`; the notebook drops it.

## Requirements

- Python 3.9+ (3.10+ recommended)
- Packages:
	- `numpy`
	- `pandas`
	- `matplotlib`
	- `seaborn`
	- `scikit-learn`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

### Option A: Run the notebook (recommended)

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open `210116_KNN.ipynb` and run all cells.

### Option B: Use the local CSV instead of the GitHub URL

The notebook currently loads the dataset from a GitHub raw URL. If you want to load the local `data.csv` (e.g., when offline), replace the data-loading cell with:

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

## What the Notebook Produces

- **Validation error vs. K** (for `k = 1..20`)
- **Best KNN model** selected via `GridSearchCV` (5-fold CV)
- **Confusion matrix** heatmap
- **ROC curve** with AUC
- **Bar chart** for Accuracy / Precision / Recall / F1 / AUC
- **PCA (2D)** visualization with a KNN decision surface

## Notes

- Scaling features is important for KNN. The notebook uses `StandardScaler` fitted on the training set, then applied to validation/test.
- The model selection uses a separate validation split (for the error curve) and also uses cross-validation in `GridSearchCV`.
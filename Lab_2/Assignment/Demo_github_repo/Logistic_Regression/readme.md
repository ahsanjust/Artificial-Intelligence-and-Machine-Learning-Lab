## Logistic Regression (Binary Classification)

### Overview
This project builds a logistic regression classifier for a binary outcome. It includes data loading, preprocessing, model training, evaluation metrics, and standard classification visualizations.

### Dataset
- Source file: `dataset/data.csv`
- Target: binary class (0/1)
- Features: numeric predictors used for classification

### Workflow
1. Load data from `dataset/data.csv`.
2. Split into train/test sets.
3. Train a logistic regression model.
4. Evaluate with accuracy, precision, recall, F1, and AUC.
5. Visualize the confusion matrix and ROC curve.

### Results (from notebook run)
- Accuracy: 0.9767441860465116
- Precision: 0.9285714285714286
- Recall: 1.0
- F1 Score: 0.9629629629629629
- AUC: 0.9993589743589744

### Visuals
The notebook generates and saves the following plots:
- Confusion matrix
- ROC curve
- Sample predictions table

### Project Files
- `210137_logistic.ipynb`: Main notebook.
- `dataset/data.csv`: Input dataset.
- `screenshots/`: Plots and tables captured from the notebook.

### Screenshots
![Confusion Matrix](screenshots/Confusion%20Matrix.png)
![Evaluation Metrics](screenshots/Evaluation%20Metrices.png)
![ROC Curve](screenshots/ROC%20Curve.png)
![Sample Predictions](screenshots/Sample%20Predictions.png)

### How to Run
1. Open `210137_logistic.ipynb` in Jupyter.
2. Ensure dependencies are available: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
3. Run all cells top to bottom.

### Notes
The metrics indicate strong class separation on the current dataset. Validate on a holdout set or with cross-validation to confirm generalization.

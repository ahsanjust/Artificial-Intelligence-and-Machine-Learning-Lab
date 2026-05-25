## Linear Regression (House Price Prediction)

### Overview
This project builds a linear regression model to predict house prices using tabular features. It covers data loading, preprocessing, model training, evaluation, and visualization of predictions and errors.

### Dataset
- Source file: `dataset/data.csv`
- Target: price (continuous)
- Features: multiple numeric inputs (e.g., sqft living and other property attributes)

### Workflow
1. Load data from `dataset/data.csv`.
2. Split into train/test sets.
3. Train a linear regression model.
4. Evaluate using MAE, MSE, and R2 score.
5. Visualize prediction errors and the regression line.

### Results (from notebook run)
- MAE: 210908.17325011527
- MSE: 986921767056.1313
- R2 Score: 0.03228385663277078

### Visuals
The notebook generates and saves the following plots:
- Loss curve / prediction errors
- Actual vs predicted prices (regression line)
- Sample predictions table

### Project Files
- `210137_linear.ipynb`: Main notebook.
- `dataset/data.csv`: Input dataset.
- `screenshots/`: Plots and tables captured from the notebook.

### Screenshots
![Evaluation Metrics](screenshots/Evaluation%20Metrics.png)
![Loss Curve](screenshots/Loss%20Curve.png)
![Actual vs Predicted Graph](screenshots/Actual%20%26%20Predicted%20Graph.png)
![Sample Predictions](screenshots/Sample%20Predictions.png)

### How to Run
1. Open `210137_linear.ipynb` in Jupyter.
2. Ensure dependencies are available: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.
3. Run all cells top to bottom.

### Notes
The low R2 suggests the current feature set or model choice may not capture the full variance in price. Consider feature scaling, outlier handling, or more expressive models for improved performance.

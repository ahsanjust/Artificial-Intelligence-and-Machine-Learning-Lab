## Linear Regression (Teens Mental Health Dataset)

### Overview
This project builds a linear regression model to predict academic performance of teens based on mental health and lifestyle factors. It covers data loading, preprocessing, model training from scratch using Gradient Descent, evaluation, and visualization.

### Dataset
- **Source file:** `dataset/Teen_Mental_Health_Dataset.csv`
- **Target:** `academic_performance` (GPA-like score, range 2.0-4.0)
- **Features:** age, gender, daily_social_media_hours, platform_usage, sleep_hours, screen_time_before_sleep, physical_activity, social_interaction_level, stress_level, anxiety_level, addiction_level

### Implementation
- **From-scratch implementation** using only NumPy (no sklearn regression models)
- **Gradient Descent** optimization with MSE cost function
- **L2 Regularization** support
- **StandardScaler** normalization for feature scaling

### Workflow
1. Load data from `dataset/Teen_Mental_Health_Dataset.csv`
2. Handle missing values and encode categorical variables
3. Split into train/test sets (80% train, 20% test)
4. Scale features using StandardScaler
5. Train Linear Regression from scratch using Gradient Descent
6. Evaluate using MAE, MSE, RMSE, and R2 Score
7. Visualize loss curve, actual vs predicted, and residual analysis

### Results
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared Score

### Visuals
The notebook generates and saves the following plots:
- Training loss curve
- Actual vs Predicted comparison
- Residual analysis

### Project Files
- `220119_linear.ipynb`: Main notebook
- `dataset/Teen_Mental_Health_Dataset.csv`: Input dataset
- `screenshots/`: Plots and tables captured from the notebook

### How to Run
1. Open `220119_linear.ipynb` in Jupyter or Google Colab
2. Ensure dependencies are available: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
3. Run all cells top to bottom

### Notes
The model learns to predict academic performance based on teens' mental health and lifestyle factors. Regularization can be adjusted to prevent overfitting.
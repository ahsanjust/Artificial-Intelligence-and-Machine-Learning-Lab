# AI & ML Lab 2: Linear and Logistic Regression (From Scratch)

**Student ID:** 220119  
**Course:** Artificial Intelligence and Machine Learning Lab  

## Overview
This repository contains solutions for the second lab assignment. The objective is to understand and implement **Linear Regression** and **Logistic Regression from scratch** using only NumPy (without sklearn's built-in model classes), and to evaluate their performance using various metrics and visualizations.

## Directory Structure
- `220119_linear.ipynb`: Jupyter Notebook with Linear Regression (from scratch implementation)
- `220119_logistic.ipynb`: Jupyter Notebook with Logistic Regression (from scratch implementation)
- `220119_links.pdf`: Document containing clickable links to GitHub and Google Colab notebooks
- `generate_notebooks.py`: Python script that generates both notebooks programmatically
- `generate_pdf.py`: Python script that generates the links PDF

---

## 1. Linear Regression (`220119_linear.ipynb`)

### Dataset
- **Teens Mental Health Dataset**
- Features: age, gender, daily_social_media_hours, platform_usage, sleep_hours, screen_time_before_sleep, physical_activity, social_interaction_level, stress_level, anxiety_level, addiction_level
- Target: academic_performance (GPA-like score)

### Implementation Details

#### From-Scratch Implementation
The Linear Regression model is implemented **using only NumPy** (no sklearn.linear_model):

```python
class LinearRegressionFromScratch:
    # Uses Gradient Descent optimization
    # L2 Regularization to prevent overfitting
    # Pure NumPy - no sklearn regressor
```

**Algorithm:**
- Hypothesis: h(x) = X * theta (matrix multiplication)
- Cost Function (MSE): J(theta) = (1/2m) * sum((h(x) - y)^2)
- Gradient: grad J(theta) = (1/m) * X^T * (X*theta - y)
- Update: theta = theta - alpha * grad J(theta)

### Workflow
1. **Data Preprocessing**:
   - Checked for missing values
   - Encoded categorical variables (gender, platform_usage, social_interaction_level)
   - Separated features (X) and target (y)
   - Split into Training (80%) and Testing (20%) sets
   - Applied StandardScaler normalization
2. **Model Training**:
   - Custom LinearRegressionFromScratch class using Gradient Descent
   - Trained for 100 epochs with loss curve visualization
3. **Visualizations**:
   - Loss Curve showing MSE decreasing over epochs
   - Actual vs Predicted scatter plot
   - Residual distribution plots
4. **Evaluation Metrics**:
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - R² Score

---

## 2. Logistic Regression (`220119_logistic.ipynb`)

### Dataset
- **Teens Mental Health Dataset**
- Features: age, gender, daily_social_media_hours, platform_usage, sleep_hours, screen_time_before_sleep, academic_performance, physical_activity, social_interaction_level, stress_level, anxiety_level, addiction_level
- Target: depression_label (0 = Not Depressed, 1 = Depressed)

### Implementation Details

#### From-Scratch Implementation
The Logistic Regression model is implemented **using only NumPy** (no sklearn.linear_model):

```python
class LogisticRegressionFromScratch:
    # Uses Sigmoid activation function
    # Binary Cross-Entropy loss
    # Gradient Descent optimization
    # L2 Regularization
    # Pure NumPy - no sklearn classifier
```

**Algorithm:**
- Hypothesis: h(x) = sigmoid(X * theta) = 1 / (1 + exp(-X*theta))
- Cost Function (Binary Cross-Entropy):
  J(theta) = -(1/m) * sum(y * log(h(x)) + (1-y) * log(1-h(x)))
- Gradient: grad J(theta) = (1/m) * X^T * (h(x) - y)
- Update: theta = theta - alpha * grad J(theta)

### Workflow
1. **Data Preprocessing**:
   - Checked for missing values
   - Encoded categorical variables (gender, platform_usage, social_interaction_level)
   - Separated features (X) and target (y)
   - Split data into 70% train, 15% validation, and 15% test sets
   - Applied StandardScaler normalization
2. **Model Training**:
   - Custom LogisticRegressionFromScratch class
   - Trained for 1000 epochs with loss curve visualization
   - Validated on validation set
3. **Evaluation Metrics (on Test Set)**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - AUC-ROC
4. **Visualizations**:
   - Training loss curve (Cross-Entropy)
   - Confusion matrix heatmap
   - ROC Curve

---

## Files Explanation

### `generate_notebooks.py`
A Python script that programmatically generates both Jupyter notebooks:
- Uses `nbformat` library to create notebook structure
- Contains complete code for both from-scratch implementations
- Adds proper markdown cells with explanations

### `generate_pdf.py`
Generates the `220119_links.pdf` file containing:
- GitHub Profile link
- Project Repository link
- Google Colab notebook links (Linear & Logistic Regression)

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Open the links provided in `220119_links.pdf`
2. Click on **Runtime** -> **Run all** to execute the cells sequentially

### Option 2: Local Environment
1. Ensure Python 3.8+ is installed
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbformat
   ```
3. Run the generate script:
   ```
   python3 generate_notebooks.py
   ```
4. Open the generated `.ipynb` files in Jupyter and run

### Option 3: Direct Notebook Execution
Simply open `220119_linear.ipynb` and `220119_logistic.ipynb` in Jupyter or Google Colab and run all cells.

---

## Summary

Both assignments demonstrate:
- Complete **from-scratch implementations** using only NumPy
- Proper **data preprocessing** pipelines
- **Gradient Descent** optimization algorithms
- Comprehensive **model evaluation** with multiple metrics
- Professional **visualizations** for analysis

The models are trained successfully on the Teens Mental Health dataset to predict academic performance (Linear Regression) and depression status (Logistic Regression).
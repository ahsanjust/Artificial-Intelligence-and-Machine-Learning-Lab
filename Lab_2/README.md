# AI & ML Lab 2: Linear and Logistic Regression (From Scratch)

**Student ID:** 220119  
**Course:** Artificial Intelligence and Machine Learning Lab

## Overview

This repository contains implementations of **Linear Regression** and **Logistic Regression from scratch** using only NumPy (without sklearn's built-in model classes). The models are trained and evaluated on the Teens Mental Health Dataset.

## Directory Structure

```
├── README.md
├── Linear_Regression/
│   ├── 220119_linear.ipynb
│   ├── dataset/
│   │   └── Teen_Mental_Health_Dataset.csv
│   └── screenshots/
├── Logistic_Regression/
│   ├── 220119_logistic.ipynb
│   ├── dataset/
│   │   └── Teen_Mental_Health_Dataset.csv
│   └── screenshots/
```

---

## 1. Linear Regression

**File:** `Linear_Regression/220119_linear.ipynb`

### Dataset
- **Source:** Teens Mental Health Dataset
- **Target:** `academic_performance` (GPA-like score, range 2.0-4.0)
- **Features:** age, gender, daily_social_media_hours, platform_usage, sleep_hours, screen_time_before_sleep, physical_activity, social_interaction_level, stress_level, anxiety_level, addiction_level

### Implementation

**From-Scratch Implementation using NumPy:**
- Gradient Descent optimization
- L2 Regularization
- MSE (Mean Squared Error) as cost function
- StandardScaler normalization

**Algorithm:**
- Hypothesis: h(x) = X × θ
- Cost: J(θ) = (1/2m) × Σ(h(x) - y)²
- Update: θ = θ - α × ∂J/∂θ

### Evaluation Metrics
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 2. Logistic Regression

**File:** `Logistic_Regression/220119_logistic.ipynb`

### Dataset
- **Source:** Teens Mental Health Dataset
- **Target:** `depression_label` (0 = Not Depressed, 1 = Depressed)
- **Features:** age, gender, daily_social_media_hours, platform_usage, sleep_hours, screen_time_before_sleep, academic_performance, physical_activity, social_interaction_level, stress_level, anxiety_level, addiction_level

### Implementation

**From-Scratch Implementation using NumPy:**
- Sigmoid activation function
- Binary Cross-Entropy loss
- Gradient Descent optimization
- L2 Regularization
- Class weights for handling imbalance (97.4% vs 2.6%)
- Optimal threshold selection using validation set

**Algorithm:**
- Hypothesis: h(x) = sigmoid(X × θ) = 1 / (1 + exp(-X×θ))
- Cost: J(θ) = -(1/m) × Σ(y × log(h(x)) + (1-y) × log(1-h(x)))
- Update: θ = θ - α × ∂J/∂θ

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

---

## How to Run

### Option 1: Local Environment
1. Ensure Python 3.8+ is installed
2. Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3. Open the notebook:
    ```bash
    jupyter notebook Linear_Regression/220119_linear.ipynb
    jupyter notebook Logistic_Regression/220119_logistic.ipynb
    ```
4. Run all cells

### Option 2: Google Colab
1. Upload the notebooks to Google Colab
2. Run all cells

---

## Summary

Both implementations demonstrate:
- Complete **from-scratch** implementations using only NumPy
- Proper **data preprocessing** pipelines
- **Gradient Descent** optimization
- Comprehensive **model evaluation** with multiple metrics
- Professional **visualizations** including loss curves, confusion matrices, and ROC curves

The models are trained on the Teens Mental Health Dataset to predict academic performance (Linear Regression) and depression status (Logistic Regression).
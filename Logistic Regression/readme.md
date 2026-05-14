## Logistic Regression (Teens Mental Health Dataset)

### Overview
This project builds a logistic regression classifier to predict depression status (binary classification) in teens based on mental health and lifestyle factors. It includes data loading, preprocessing, model training from scratch using Gradient Descent, evaluation metrics, and standard classification visualizations.

### Dataset
- **Source file:** `dataset/Teen_Mental_Health_Dataset.csv`
- **Target:** `depression_label` (0 = Not Depressed, 1 = Depressed)
- **Features:** age, gender, daily_social_media_hours, platform_usage, sleep_hours, screen_time_before_sleep, academic_performance, physical_activity, social_interaction_level, stress_level, anxiety_level, addiction_level

### Implementation
- **From-scratch implementation** using only NumPy (no sklearn logistic regression)
- **Sigmoid activation** function
- **Binary Cross-Entropy** loss function
- **Gradient Descent** optimization
- **L2 Regularization** support
- **Class weights** to handle class imbalance
- **Optimal threshold selection** using validation set

### Workflow
1. Load data from `dataset/Teen_Mental_Health_Dataset.csv`
2. Handle missing values and encode categorical variables
3. Split into train/validation/test sets (70% train, 15% val, 15% test)
4. Scale features using StandardScaler
5. Train Logistic Regression from scratch using Gradient Descent
6. Optimize classification threshold using validation set
7. Evaluate with accuracy, precision, recall, F1-score, and AUC-ROC
8. Visualize confusion matrix and ROC curve

### Results
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC Score

### Visuals
The notebook generates and saves the following plots:
- Training loss curve
- Confusion matrix
- ROC curve
- Sample predictions with probabilities

### Project Files
- `220119_logistic.ipynb`: Main notebook
- `dataset/Teen_Mental_Health_Dataset.csv`: Input dataset
- `screenshots/`: Plots and tables captured from the notebook

### How to Run
1. Open `220119_logistic.ipynb` in Jupyter or Google Colab
2. Ensure dependencies are available: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
3. Run all cells top to bottom

### Notes
The model learns to predict depression status based on teens' mental health and lifestyle factors. Class weights are used to handle the imbalanced dataset.
import nbformat as nbf
import os


def create_linear_regression_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# Assignment 1: Linear Regression (From Scratch)\n\n"
        "This notebook implements **Linear Regression from scratch** using only NumPy, without using scikit-learn's built-in regressor. "
        "We use Gradient Descent optimization to find the optimal weights that minimize the Mean Squared Error (MSE).\n\n"
        "## Dataset\n- **California Housing Dataset** (via scikit-learn)\n- Target: Median house value\n- Features: Median income, house age, average rooms, etc."
    ))

    code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"""
    cells.append(nbf.v4.new_code_cell(code_imports))

    cells.append(nbf.v4.new_markdown_cell("## 1. Load and Explore the Dataset"))
    code_load = """# Load California Housing Dataset
california = fetch_california_housing(as_frame=True)
df = california.frame
print("Dataset Shape:", df.shape)
print("\\nFirst 5 rows:")
df.head()"""
    cells.append(nbf.v4.new_code_cell(code_load))

    cells.append(nbf.v4.new_markdown_cell("## 2. Data Preprocessing\n\n"
        "### 2.1 Check for Missing Values\nWe verify there are no missing values in the dataset."))
    code_missing = """# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print(f"\\nTotal missing values: {df.isnull().sum().sum()}")"""
    cells.append(nbf.v4.new_code_cell(code_missing))

    cells.append(nbf.v4.new_markdown_cell("### 2.2 Feature Selection and Train-Test Split\n"
        "We separate features (X) and target (y), then split into 80% training and 20% testing sets."))
    code_split = """# Feature Selection
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")"""
    cells.append(nbf.v4.new_code_cell(code_split))

    cells.append(nbf.v4.new_markdown_cell("### 2.3 Feature Scaling (Standard Normalization)\n"
        "We apply StandardScaler to normalize features. This is crucial for gradient descent to converge properly."))
    code_scale = """# Normalization using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("After scaling:")
print(f"X_train mean (should be ~0): {X_train_scaled.mean(axis=0)[:3]}")
print(f"X_train std (should be ~1): {X_train_scaled.std(axis=0)[:3]}")"""
    cells.append(nbf.v4.new_code_cell(code_scale))

    cells.append(nbf.v4.new_markdown_cell("## 3. Linear Regression Implementation (From Scratch)\n\n"
        "### 3.1 Theory\n\n"
        "Linear Regression using Gradient Descent:\n"
        "- Hypothesis: h(x) = X * theta (matrix multiplication)\n"
        "- Cost Function (MSE): J(theta) = (1/2m) * sum((h(x) - y)^2)\n"
        "- Gradient: grad J(theta) = (1/m) * X^T * (X*theta - y)\n"
        "- Update: theta = theta - alpha * grad J(theta)\n\n"
        "where alpha is the learning rate."))

    code_linear_scratch = "class LinearRegressionFromScratch:\n    \"\"\"\n    Linear Regression implemented from scratch using Gradient Descent.\n    This implementation does NOT use sklearn SGDRegressor - it uses pure NumPy.\n    \"\"\"\n    def __init__(self, learning_rate=0.01, n_epochs=100, regularization=0.01):\n        self.learning_rate = learning_rate\n        self.n_epochs = n_epochs\n        self.regularization = regularization\n        self.theta = None\n        self.loss_history = []\n    \n    def _add_bias(self, X):\n        return np.c_[np.ones(X.shape[0]), X]\n    \n    def fit(self, X, y):\n        m, n = X.shape\n        X_b = self._add_bias(X)\n        self.theta = np.zeros(n + 1)\n        \n        for epoch in range(self.n_epochs):\n            predictions = X_b @ self.theta\n            error = predictions - y\n            gradient = (1/m) * (X_b.T @ error)\n            gradient[1:] += (self.regularization / m) * self.theta[1:]\n            self.theta -= self.learning_rate * gradient\n            mse = (1/(2*m)) * np.sum(error**2)\n            self.loss_history.append(mse)\n        return self\n    \n    def predict(self, X):\n        X_b = self._add_bias(X)\n        return X_b @ self.theta\n\nprint(\"LinearRegressionFromScratch class defined!\")"
    cells.append(nbf.v4.new_code_cell(code_linear_scratch))

    cells.append(nbf.v4.new_markdown_cell("### 3.2 Train the Model\n\n"
        "We train our from-scratch model with appropriate hyperparameters:\n"
        "- Learning rate: 0.01 (small enough for stable convergence)\n"
        "- Epochs: 100 (to visualize the loss curve)\n"
        "- Regularization: 0.01 (L2 penalty)"))

    code_train = """# Create and train the model
model = LinearRegressionFromScratch(
    learning_rate=0.01, 
    n_epochs=100, 
    regularization=0.01
)

# Train on scaled training data
model.fit(X_train_scaled, y_train)

print(f"Training completed!")
print(f"Final training MSE: {model.loss_history[-1]:.4f}")
print(f"Number of epochs: {len(model.loss_history)}")"""
    cells.append(nbf.v4.new_code_cell(code_train))

    cells.append(nbf.v4.new_markdown_cell("### 3.3 Visualize Loss Curve\n\n"
        "The loss curve shows how the MSE decreases over epochs, proving the model is learning."))

    code_loss_curve = """# Plot Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(range(model.n_epochs), model.loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('Training Loss Curve - Linear Regression (From Scratch)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\\nObservations:")
print(f"- Initial MSE: {model.loss_history[0]:.4f}")
print(f"- Final MSE: {model.loss_history[-1]:.4f}")
print(f"- MSE Reduction: {((model.loss_history[0] - model.loss_history[-1])/model.loss_history[0]*100):.2f}%")"""
    cells.append(nbf.v4.new_code_cell(code_loss_curve))

    cells.append(nbf.v4.new_markdown_cell("## 4. Model Evaluation\n\n"
        "We evaluate the model using multiple metrics on the test set."))

    code_eval = """# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("LINEAR REGRESSION MODEL EVALUATION (From Scratch)")
print("=" * 50)
print(f"Mean Absolute Error (MAE):       {mae:.4f}")
print(f"Mean Squared Error (MSE):        {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2):                   {r2:.4f}")
print("=" * 50)

if r2 > 0.7:
    print("\\nGood model fit! The model explains more than 70% of variance.")
elif r2 > 0.5:
    print("\\nModerate model fit. Consider tuning hyperparameters.")
else:
    print("\\nPoor model fit. Try adjusting learning rate or epochs.")"""
    cells.append(nbf.v4.new_code_cell(code_eval))

    cells.append(nbf.v4.new_markdown_cell("### 4.1 Actual vs Predicted Comparison\n\n"
        "Let's compare some actual values with predictions."))

    code_sample = """# Sample predictions comparison
sample_indices = range(10)
sample_df = pd.DataFrame({
    'Actual Value': y_test.iloc[sample_indices].values,
    'Predicted Value': y_pred[sample_indices],
    'Error': y_test.iloc[sample_indices].values - y_pred[sample_indices]
})
print("Sample Predictions vs Actual Values:")
display(sample_df)

# Scatter plot of Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Actual vs Predicted Values', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""
    cells.append(nbf.v4.new_code_cell(code_sample))

    cells.append(nbf.v4.new_markdown_cell("### 4.2 Residual Analysis\n\n"
        "Residuals (errors) should be normally distributed around zero for a good model."))

    code_residual = """# Residual analysis
residuals = y_test.values - y_pred

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residual (Actual - Predicted)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of Residuals', fontsize=12)
plt.axvline(x=0, color='r', linestyle='--', label='Zero')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.3, edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values', fontsize=11)
plt.ylabel('Residuals', fontsize=11)
plt.title('Residual Plot', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Mean Residual: {np.mean(residuals):.4f} (should be close to 0)")
print(f"Std of Residuals: {np.std(residuals):.4f}")"""
    cells.append(nbf.v4.new_code_cell(code_residual))

    cells.append(nbf.v4.new_markdown_cell("## 5. Summary\n\n"
        "This notebook demonstrated:\n"
        "1. **Data Preprocessing**: Handling missing values, train-test splitting, and feature normalization\n"
        "2. **From-Scratch Implementation**: Linear Regression using pure NumPy with Gradient Descent\n"
        "3. **Model Evaluation**: MAE, MSE, RMSE, R2 metrics\n"
        "4. **Visualization**: Loss curve, actual vs predicted, residual analysis\n\n"
        "The model successfully learns to predict California housing prices."))

    nb['cells'] = cells
    return nb


def create_logistic_regression_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# Assignment 2: Logistic Regression (From Scratch)\n\n"
        "This notebook implements **Logistic Regression from scratch** using only NumPy, without using scikit-learn's built-in classifier. "
        "We use Gradient Descent optimization with the binary cross-entropy loss function.\n\n"
        "## Dataset\n- **Breast Cancer Wisconsin Dataset** (via scikit-learn)\n- Target: Binary classification (Malignant=0, Benign=1)\n- Features: 30 measurements of cell nuclei characteristics"
    ))

    code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve"""
    cells.append(nbf.v4.new_code_cell(code_imports))

    cells.append(nbf.v4.new_markdown_cell("## 1. Load and Explore the Dataset"))

    code_load = """# Load Breast Cancer Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

print("Dataset Shape:", df.shape)
print(f"Target classes: {data.target_names}")
print(f"Class distribution: {np.bincount(data.target)}")
print("\\nFirst 5 rows:")
df.head()"""
    cells.append(nbf.v4.new_code_cell(code_load))

    cells.append(nbf.v4.new_markdown_cell("## 2. Data Preprocessing\n\n"
        "### 2.1 Introduce Missing Values (for demonstration)\n"
        "We intentionally introduce missing values to demonstrate handling techniques."))

    code_missing_intro = """# Introduce missing values for demonstration purposes
np.random.seed(42)
df_with_nan = df.copy()
df_with_nan.loc[0:10, 'mean radius'] = np.nan

print("Missing values (after introducing):")
print(df_with_nan.isnull().sum().head())"""
    cells.append(nbf.v4.new_code_cell(code_missing_intro))

    cells.append(nbf.v4.new_markdown_cell("### 2.2 Handle Missing Values\n"
        "We impute missing values using the column mean."))

    code_impute = """# Handle missing values - Imputation with mean
df_imputed = df_with_nan.copy()
df_imputed['mean radius'] = df_imputed['mean radius'].fillna(df_imputed['mean radius'].mean())

print("Missing values after imputation:", df_imputed.isnull().sum().sum())"""
    cells.append(nbf.v4.new_code_cell(code_impute))

    cells.append(nbf.v4.new_markdown_cell("### 2.3 Introduce and Encode Categorical Feature\n"
        "We add a synthetic categorical feature and demonstrate label encoding."))

    code_categorical = """# Introduce categorical feature
np.random.seed(42)
df_imputed['category_feature'] = np.random.choice(['Type_A', 'Type_B', 'Type_C'], size=len(df_imputed))

# Encode categorical feature using LabelEncoder
encoder = LabelEncoder()
df_imputed['category_feature'] = encoder.fit_transform(df_imputed['category_feature'])

print("Categorical encoding:")
for i, label in enumerate(encoder.classes_):
    print(f"  {label} -> {i}")"""
    cells.append(nbf.v4.new_code_cell(code_categorical))

    cells.append(nbf.v4.new_markdown_cell("### 2.4 Train-Validation-Test Split\n"
        "We split data into 70% train, 15% validation, and 15% test sets."))

    code_split = """# Prepare features and target
df_imputed['target'] = data.target
X = df_imputed.drop('target', axis=1)
y = df_imputed['target']

# Split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Split temp: 50% validation, 50% test (15% each of total)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training samples:   {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation samples: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test samples:       {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")"""
    cells.append(nbf.v4.new_code_cell(code_split))

    cells.append(nbf.v4.new_markdown_cell("### 2.5 Feature Normalization\n"
        "Apply StandardScaler to normalize features for gradient descent."))

    code_scale = """# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")
print(f"X_train_scaled mean: {X_train_scaled.mean():.6f}")
print(f"X_train_scaled std:  {X_train_scaled.std():.6f}")"""
    cells.append(nbf.v4.new_code_cell(code_scale))

    cells.append(nbf.v4.new_markdown_cell("## 3. Logistic Regression Implementation (From Scratch)\n\n"
        "### 3.1 Theory\n\n"
        "Logistic Regression uses the sigmoid function:\n"
        "- Hypothesis: h(x) = sigmoid(X * theta) = 1 / (1 + exp(-X*theta))\n"
        "- Cost Function (Binary Cross-Entropy):\n"
        "  J(theta) = -(1/m) * sum(y * log(h(x)) + (1-y) * log(1-h(x)))\n"
        "- Gradient: grad J(theta) = (1/m) * X^T * (h(x) - y)\n"
        "- Update: theta = theta - alpha * grad J(theta)"))

    code_logistic_scratch = "class LogisticRegressionFromScratch:\n    \"\"\"\n    Logistic Regression implemented from scratch using Gradient Descent.\n    This implementation does NOT use sklearn LogisticRegression - it uses pure NumPy.\n    \"\"\"\n    def __init__(self, learning_rate=0.1, n_epochs=1000, regularization=0.01):\n        self.learning_rate = learning_rate\n        self.n_epochs = n_epochs\n        self.regularization = regularization\n        self.theta = None\n        self.loss_history = []\n    \n    def _sigmoid(self, z):\n        z = np.clip(z, -500, 500)\n        return 1 / (1 + np.exp(-z))\n    \n    def _add_bias(self, X):\n        return np.c_[np.ones(X.shape[0]), X]\n    \n    def fit(self, X, y):\n        m, n = X.shape\n        X_b = self._add_bias(X)\n        self.theta = np.zeros(n + 1)\n        \n        for epoch in range(self.n_epochs):\n            z = X_b @ self.theta\n            predictions = self._sigmoid(z)\n            error = predictions - y\n            gradient = (1/m) * (X_b.T @ error)\n            gradient[1:] += (self.regularization / m) * self.theta[1:]\n            self.theta -= self.learning_rate * gradient\n            predictions = np.clip(predictions, 1e-10, 1 - 1e-10)\n            loss = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))\n            self.loss_history.append(loss)\n        return self\n    \n    def predict_proba(self, X):\n        X_b = self._add_bias(X)\n        return self._sigmoid(X_b @ self.theta)\n    \n    def predict(self, X, threshold=0.5):\n        return (self.predict_proba(X) >= threshold).astype(int)\n\nprint(\"LogisticRegressionFromScratch class defined!\")"
    cells.append(nbf.v4.new_code_cell(code_logistic_scratch))

    cells.append(nbf.v4.new_markdown_cell("### 3.2 Train the Model"))

    code_train = """# Create and train the model
log_model = LogisticRegressionFromScratch(
    learning_rate=0.1,
    n_epochs=1000,
    regularization=0.01
)

# Train
log_model.fit(X_train_scaled, y_train)

print(f"Training completed!")
print(f"Final training loss (Cross-Entropy): {log_model.loss_history[-1]:.4f}")
print(f"Total epochs: {len(log_model.loss_history)}")"""
    cells.append(nbf.v4.new_code_cell(code_train))

    cells.append(nbf.v4.new_markdown_cell("### 3.3 Training Loss Curve"))

    code_loss_curve = """# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(log_model.n_epochs), log_model.loss_history, 'b-', linewidth=1.5)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
plt.title('Training Loss Curve - Logistic Regression (From Scratch)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Initial loss: {log_model.loss_history[0]:.4f}")
print(f"Final loss: {log_model.loss_history[-1]:.4f}")
print(f"Loss reduction: {((log_model.loss_history[0] - log_model.loss_history[-1])/log_model.loss_history[0]*100):.2f}%")"""
    cells.append(nbf.v4.new_code_cell(code_loss_curve))

    cells.append(nbf.v4.new_markdown_cell("### 3.4 Validation\n"
        "Check performance on the validation set to detect overfitting."))

    code_validate = """# Validate on validation set
y_val_pred = log_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.4f}")

# Also check training accuracy for comparison
y_train_pred = log_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

if val_accuracy < train_accuracy - 0.05:
    print("\\nWarning: Possible overfitting (validation much lower than training)")
else:
    print("\\nNo significant overfitting detected")"""
    cells.append(nbf.v4.new_code_cell(code_validate))

    cells.append(nbf.v4.new_markdown_cell("## 4. Model Evaluation on Test Set"))

    code_eval = """# Generate predictions on test set
y_pred = log_model.predict(X_test_scaled)
y_prob = log_model.predict_proba(X_test_scaled)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 55)
print("LOGISTIC REGRESSION MODEL EVALUATION (From Scratch)")
print("=" * 55)
print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision:    {prec:.4f}")
print(f"Recall:       {rec:.4f}")
print(f"F1-Score:     {f1:.4f}")
print(f"AUC-ROC:      {auc:.4f}")
print("=" * 55)"""
    cells.append(nbf.v4.new_code_cell(code_eval))

    cells.append(nbf.v4.new_markdown_cell("### 4.1 Confusion Matrix"))

    code_cm = """# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant (0)', 'Benign (1)'],
            yticklabels=['Malignant (0)', 'Benign (1)'],
            annot_kws={'size': 14})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Malignant correct):  {tn}")
print(f"False Positives (Benign wrong):       {fp}")
print(f"False Negatives (Malignant wrong):    {fn}")
print(f"True Positives (Benign correct):      {tp}")"""
    cells.append(nbf.v4.new_code_cell(code_cm))

    cells.append(nbf.v4.new_markdown_cell("### 4.2 ROC Curve"))

    code_roc = """# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Logistic Regression (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"AUC = {auc:.4f}")
print("AUC of 1.0 = Perfect classifier")
print("AUC of 0.5 = Random classifier")"""
    cells.append(nbf.v4.new_code_cell(code_roc))

    cells.append(nbf.v4.new_markdown_cell("### 4.3 Sample Predictions with Probabilities"))

    code_sample = """# Sample predictions with probabilities
sample_df = pd.DataFrame({
    'Actual': y_test[:10].values,
    'Predicted': y_pred[:10],
    'Probability (Benign)': y_prob[:10]
})
sample_df['Actual Label'] = sample_df['Actual'].map({0: 'Malignant', 1: 'Benign'})
sample_df['Predicted Label'] = sample_df['Predicted'].map({0: 'Malignant', 1: 'Benign'})

print("Sample Predictions with Probability Scores:")
display(sample_df[['Actual Label', 'Predicted Label', 'Probability (Benign)']])

correct = (y_pred == y_test.values).sum()
print(f"\\nCorrect predictions: {correct}/{len(y_test)} ({correct/len(y_test)*100:.1f}%)")"""
    cells.append(nbf.v4.new_code_cell(code_sample))

    cells.append(nbf.v4.new_markdown_cell("## 5. Summary\n\n"
        "This notebook demonstrated:\n"
        "1. **Complete Preprocessing**: Missing value imputation, categorical encoding, train/val/test split, feature normalization\n"
        "2. **From-Scratch Implementation**: Logistic Regression using pure NumPy with Gradient Descent\n"
        "3. **Binary Classification**: Using sigmoid function and cross-entropy loss\n"
        "4. **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, AUC-ROC\n"
        "5. **Visualization**: Training loss curve, confusion matrix, ROC curve\n\n"
        "The model achieves high accuracy in detecting breast cancer cases."))

    nb['cells'] = cells
    return nb


if __name__ == "__main__":
    output_dir = "Lab 2"
    os.makedirs(output_dir, exist_ok=True)
    
    nb_linear = create_linear_regression_notebook()
    nbf.write(nb_linear, '220119_linear.ipynb')
    print("Linear Regression notebook created: 220119_linear.ipynb")
    
    nb_logistic = create_logistic_regression_notebook()
    nbf.write(nb_logistic, '220119_logistic.ipynb')
    print("Logistic Regression notebook created: 220119_logistic.ipynb")
    
    print("\n" + "="*50)
    print("All notebooks generated successfully!")
    print("="*50)
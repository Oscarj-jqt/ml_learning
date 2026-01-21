# Introduction to Machine Learning

## What is a Model?
A **model** in machine learning is like a smart recipe that learns patterns from data.  
- It takes input data (features) and tries to predict an output (target).
- Example: Predicting house prices based on features like size, year built, etc.

## Example: Basic ML Workflow in Python

```python
# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Load data
data = pd.read_csv("my_data.csv")

# Select features and target
X = data[["LotArea", "YearBuilt", "1stFlrSF"]]
y = data["SalePrice"]

# Define and fit the model
model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Show first predictions and actual values
print(predictions[:5])
print(y.head())
```

## Why Do We Use Models?
- To make predictions or decisions automatically, based on data.
- To find patterns that are too complex for humans to spot easily.

## The Main Steps in a Machine Learning Project

### 1. Define
- Decide what kind of model to use (e.g., Decision Tree, Linear Regression).
- Set parameters for the model (e.g., random_state for reproducibility).

### 2. Fit
- “Fitting” means training the model: the model looks at the data and learns the relationship between features (X) and the target (y).
- This is where the model “learns” from the data.

### 3. Predict
- After training, the model can make predictions on new or existing data.
- Example:
```python
predictions = model.predict(X)
```

### 4. Evaluate
- Check how good the model’s predictions are.
- Compare predictions to actual values (e.g., using accuracy, mean absolute error, etc.).
- Helps you know if your model is useful or needs improvement.

## Key Concepts

- **Features (X):** The input variables (columns) used to make predictions.
- **Target (y):** The value you want to predict.
- **Overfitting:** When a model memorizes the training data and doesn’t generalize well to new data.
- **Validation:** Testing your model on data it hasn’t seen before to check if it works well in real situations.

## Why Do We Fit a Model?
- To let the model “learn” from the data so it can make accurate predictions.

## What Are Predictions?
- The model’s guesses for the target values, based on the patterns it learned during training.
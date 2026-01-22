# Model Validation

Model validation is meant to compare predictions with real values. Prediction errors represent the uncertainty caused by incorrect predictions. A common metric to measure this error is **MAE** (Mean Absolute Error).

## Example: Regression Model

After training the model, here are the typical steps:

```python
import pandas as pd

# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Filter rows to ensure there will be no missing values for price
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor

# Define and fit the model
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(X, y)
```

### Calculating Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mae = mean_absolute_error(y, predicted_home_prices)
print(mae)
```

> **Note:**  
> Here, MAE is calculated on the same data used for training. This can give an overly optimistic error, since the model has already "seen" these data.

## Why Use Validation Data?

Problems can occur if the model encounters features it was not trained on. To evaluate real performance, we use **validation data**: data the model has never seen during training.

### Splitting Data with `train_test_split`

```python
from sklearn.model_selection import train_test_split

# Split data into training and validation sets for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define and fit the model on training data
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

# Predict on validation data
val_predictions = melbourne_model.predict(val_X)
mae_val = mean_absolute_error(val_y, val_predictions)
print(mae_val)
```

> **Note:**  
> - The `random_state` argument ensures you get the same split every time you run the script.
> - MAE calculated on validation data gives a more realistic estimate of model performance on new, unseen data.

## Key Points

- **Validation** = testing the model on data it has never seen.
- **MAE** measures the average error between actual values and predictions.
- Always split your data into training and validation sets.
- Use `train_test_split` for a random and reproducible split.
# Model Fitting: Overfitting, Underfitting, and Model Selection

## Overfitting and Underfitting

- **Overfitting**:  
  This happens when a model matches the training data almost perfectly, but performs poorly on validation or new data. The model has "memorized" the training data instead of learning general patterns.

- **Underfitting**:  
  This is when a model does not capture important distinctions and patterns in the data. It performs badly even on the training data, meaning it is too simple to represent the underlying relationships.

---

## Comparing Model Complexity

To find the best model, we can compare the Mean Absolute Error (MAE) for different values of `max_leaf_nodes` in a Decision Tree.

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

> **Note:**  
> The function `get_mae` helps you compare how well different tree sizes perform on validation data.  
> - A small tree (few leaves) may underfit.  
> - A very large tree may overfit.

---

## Loop to Compare Different Tree Sizes

We can use a loop to test several values for `max_leaf_nodes` and see which gives the lowest MAE.

```python
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
mae_dict = {}

for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    mae_dict[max_leaf_nodes] = my_mae
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))

# Find the optimal tree size
best_tree_size = min(mae_dict, key=mae_dict.get)

## Brief Description of Models

### Decision Tree

A **Decision Tree** is a model that makes predictions by learning simple decision rules from the data. It splits the data into branches based on feature values, like answering a series of yes/no questions, until it reaches a prediction.

- **Why use it?**  
  It’s easy to understand and interpret. Good for simple problems or as a starting point.

- **Limitations:**  
  Decision trees can easily overfit (memorize) the training data and perform poorly on new data.

---

### Random Forest

A **Random Forest** is an ensemble of many decision trees. Each tree is trained on a random subset of the data and features. The final prediction is made by averaging (for regression) or voting (for classification) the results of all trees.

- **Why use it?**  
  It reduces overfitting and usually gives better, more stable results than a single decision tree.

- **Limitations:**  
  More complex and slower than a single tree, but much more accurate in most cases.

---

### Gradient Boosting

**Gradient Boosting** builds a series of decision trees, where each new tree tries to correct the errors made by the previous ones. The predictions are combined to make a final, improved prediction.

- **Why use it?**  
  It often achieves the best performance, especially on complex datasets. It can capture subtle patterns in the data.

- **Limitations:**  
  More complex and can take longer to train. Needs careful tuning to avoid overfitting.

---

**Summary:**  
- **Decision Tree:** Simple, easy to interpret, but can overfit.
- **Random Forest:** Many trees, less overfitting, more accurate.
- **Gradient Boosting:** Trees correct each other, often best accuracy, but more complex.

We use these models to find the best balance between simplicity, accuracy, and the ability to generalize to new data.
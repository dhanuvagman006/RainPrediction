

### Display the count of unique values in the "rainfall" column of the df_downsampled DataFrame.


```python
df_downsampled.head()
```  
This line counts the unique values in the "rainfall" column of the df_downsampled DataFrame, which can help understand the distribution of the target variable
### Split the DataFrame into features (X) and target variable (y).
```python
X = df_downsampled.drop(columns=["rainfall"]) 
``` 
Drop the "rainfall" column to create the feature set.
```python
y = df_downsampled["rainfall"]  
``` 
Assign the "rainfall" column as the target variable.

### Print the features and target variable to verify the split.
```python
print(X)  
```
 Display the feature set.
```python
print(y)  
```
 Display the target variable.

```python
X_train 
```
 X_train: Features for training.
```python
 X_test
 ```
  X_test: Features for testing.
  ```python
  y_train 
  ```
  y_train: Target variable for training.

```python
   y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 ```
  y_test: Target variable for testing.


  
 


### Define a parameter grid for hyperparameter tuning of the Random Forest model.
```python
param_grid_rf = {
    "n_estimators": [50, 100, 200],  # Number of trees in the forest.
    "max_features": ["sqrt", "log2"],  # Number of features to consider when looking for the best split.
    "max_depth": [None, 10, 20, 30],  # Maximum depth of the tree.
    "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node.
    "min_samples_leaf": [1, 2, 4]  # Minimum number of samples required to be at a leaf node.
}
```
A parameter grid is defined for hyperparameter tuning. This grid specifies different values for various hyperparameters of the Random Forest model

### Initialize GridSearchCV for hyperparameter tuning with cross-validation.

```python
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
```
 GridSearchCV is initialized to perform hyperparameter tuning with cross-validation. The cv parameter specifies the number of folds for cross-validation, and n_jobs=-1 allows the use of all available processors


### Fit the GridSearchCV to the training data to find the best hyperparameters.

```python
grid_search_rf.fit(X_train, y_train)
```
The GridSearchCV is fitted to the training data to find the best hyperparameters

### Extract the best estimator (model) found during the grid search.

```python
best_rf_model = grid_search_rf.best_estimator_
```
The best model found during the grid search is extracted for further use

### Print the best hyperparameters found for the Random Forest model.
```python
print("best parameters for Random Forest:", grid_search_rf.best_params_)
```
the best hyperparameters found for the Random Forest model are print




















































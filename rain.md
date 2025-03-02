# Rain Prediction

This project aims to predict rainfall using machine learning techniques. It leverages Python libraries for data processing, visualization, and model training.

## Dependencies
Ensure you have the following Python libraries installed:

- `numpy` – Numerical computations
- `pandas` – Data manipulation and analysis
- `matplotlib` – Data visualization
- `seaborn` – Statistical data visualization
- `scikit-learn` – Machine learning tools
- `pickle` – Model saving and loading

### Installation
Install the required libraries using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
## Importing Libraries
```python
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

from sklearn.utils import resample  
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
import pickle  
```


#Display the count of unique values in the "rainfall" column of the df_downsampled DataFrame.

df_downsampled.head()
```
`df_downsampled.head()`This line counts the unique values in the "rainfall" column of the df_downsampled DataFrame, which can help understand the distribution of the target variable.
```
# Split the DataFrame into features (X) and target variable (y).


X = df_downsampled.drop(columns=["rainfall"]) 
```
`X = df_downsampled.drop(columns=["rainfall"])` Drop the "rainfall" column to create the feature set. 
```
 
y = df_downsampled["rainfall"] 
```
`y = df_downsampled["rainfall"]`Assign the "rainfall" column as the target variable.
 ``` 

# Print the features and target variable to verify the split.


print(X)
``` 
`print(X)`  Display the feature set.
```
  
print(y)
```
`print(y)`Display the target variable.
```

# Split the data into training and testing sets using an 80-20 split.

X_train
```
`X_train`X_train: Features for training.
```
 
X_test
```
`X_test` X_test: Features for testing.
```

 y_train
 ```
 `y_train`y_train: Target variable for training.
 ```
 y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 ```
` y_test = train_test_split(X, y, test_size=0.2, random_state=42)` y_test: Target variable for testing.
```
  
 
# Initialize a Random Forest Classifier with a specified random state for reproducibility.

rf_model = RandomForestClassifier(random_state=42)
```
`rf_model = RandomForestClassifier(random_state=42)`A Random Forest Classifier is initialized with a specified random state for reproducibility.
```    

# Define a parameter grid for hyperparameter tuning of the Random Forest model.



param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
```
`param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]}`A parameter grid is defined for hyperparameter tuning. This grid specifies different values for various hyperparameters of the Random Forest model

```
# Initialize GridSearchCV for hyperparameter tuning with cross-validation.

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
```
`grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)`GridSearchCV is initialized to perform hyperparameter tuning with cross-validation. The cv parameter specifies the number of folds for cross-validation, and n_jobs=-1 allows the use of all available processors.
```

# Fit the GridSearchCV to the training data to find the best hyperparameters.

grid_search_rf.fit(X_train, y_train)
```
`grid_search_rf.fit(X_train, y_train)`The GridSearchCV is fitted to the training data to find the best hyperparameters.
```

# Extract the best estimator (model) found during the grid search.

best_rf_model = grid_search_rf.best_estimator_
```
`best_rf_model = grid_search_rf.best_estimator_`The best model found during the grid search is extracted for further use.
```    

# Print the best hyperparameters found for the Random Forest model.

print("best parameters for Random Forest:", grid_search_rf.best_params_)
```
`print("best parameters for Random Forest:", grid_search_rf.best_params_)` the best hyperparameters found for the Random Forest model are printe
 ```   
# Display the count of unique values in the "rainfall" column of the df_downsampled DataFrame.
df_downsampled["rainfall"].value_counts()

# Split the DataFrame into features (X) and target variable (y).
X = df_downsampled.drop(columns=["rainfall"])  # Drop the "rainfall" column to create the feature set.
y = df_downsampled["rainfall"]  # Assign the "rainfall" column as the target variable.

# Print the features and target variable to verify the split.
print(X)  # Display the feature set.
print(y)  # Display the target variable.

# Split the data into training and testing sets using an 80-20 split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train: Features for training.
# X_test: Features for testing.
# y_train: Target variable for training.
# y_test: Target variable for testing.

# Initialize a Random Forest Classifier with a specified random state for reproducibility.
rf_model = RandomForestClassifier(random_state=42)

# Define a parameter grid for hyperparameter tuning of the Random Forest model.
param_grid_rf = {
    "n_estimators": [50, 100, 200],  # Number of trees in the forest.
    "max_features": ["sqrt", "log2"],  # Number of features to consider when looking for the best split.
    "max_depth": [None, 10, 20, 30],  # Maximum depth of the tree.
    "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node.
    "min_samples_leaf": [1, 2, 4]  # Minimum number of samples required to be at a leaf node.
}

# Initialize GridSearchCV for hyperparameter tuning with cross-validation.
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
# estimator: The model to be tuned (Random Forest).
# param_grid: The parameter grid to search over.
# cv: Number of cross-validation folds.
# n_jobs: Number of jobs to run in parallel (-1 means using all processors).
# verbose: Controls the verbosity of the output during the fitting process.

# Fit the GridSearchCV to the training data to find the best hyperparameters.
grid_search_rf.fit(X_train, y_train)

# Extract the best estimator (model) found during the grid search.
best_rf_model = grid_search_rf.best_estimator_

# Print the best hyperparameters found for the Random Forest model.
print("best parameters for Random Forest:", grid_search_rf.best_params_)
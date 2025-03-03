# Rain Prediction

This project aims to predict rainfall using machine learning techniques. It leverages Python libraries for data processing, visualization, and model training.

## Dependencies

Ensure you have the following Python libraries installed:
- `numpy` 
- `pandas` 
- `matplotlib` 
- `seaborn` 
- `scikit-learn` 
- `pickle` 

You can install them using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

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

`import numpy as np` imports the NumPy library, which is used for numerical computations in Python.\
`import pandas as pd` imports the Pandas library, which is used for data analysis and manipulation in Python.\
`import matplotlib.pyplot as plt` imports Matplotlib’s Pyplot module for creating visualizations like graphs and charts.\
`import seaborn as sns` imports Seaborn, a Python library for making statistical data visualizations.\
`from sklearn.utils import resample` imports the `resample` function from **scikit-learn**, which is used for **random sampling, bootstrapping, and data balancing** .\
`from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score` imports functions for **data splitting, hyperparameter tuning, and cross-validation**.\
`from sklearn.ensemble import RandomForestClassifier` imports the **RandomForestClassifier** for **building ensemble-based classification models**.\
`from sklearn.metrics import classification_report, confusion_matrix, accuracy_score` imports functions for **evaluating model performance**.\
`import pickle` imports the **Pickle** module for **saving and loading Python objects**.\

```
data = pd.read_csv("/content/Rainfall.csv")
```
`data = pd.read_csv("/content/Rainfall.csv")` loads the **"Rainfall.csv"** file into a Pandas DataFrame for data analysis.

```
data.head()
```
`data.head()` displays the first **five rows** of the dataset.

```
# remove extra  spaces in all columns
data.columns = data.columns.str.strip()
```
`data.columns = data.columns.str.strip()` removes **extra spaces** from all column names to ensure clean and consistent headers.


```
data.columns
```
`data.columns` displays the **list of column names** in the dataset.

```
print("Data Info:")
data.info()
```
`print("Data Info:")` simply prints the text **"Data Info:"** to the output.\
`data.info()` displays **column names, data types, non-null counts, and memory usage** of the dataset.

```
data = data.drop(columns=["day"])
```
`data = data.drop(columns=["day"])` removes the **"day"** column from the dataset.

```
data.head()
```
`data.head()` displays the first **five rows** of the dataset.

```
print(data.isnull().sum())
```
`print(data.isnull().sum())` prints the **number of missing (null) values** in each column of the dataset.

```
data["winddirection"].unique()
```
`data["winddirection"].unique()` returns an array of **unique values** present in the `"winddirection"` column.

```
# handle missing values
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())
```
`data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])` **fills missing values** in the `"winddirection"` column with the most frequently occurring value (**mode**).\
`data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())` **fills missing values** in the `"windspeed"` column with its **median** value.

### checking the number of missing values


```
print(data.isnull().sum()) 
```
prints the number of missing values in each coloum of the data set.


```
data["rainfall"].unique() 
```
The command `data["rainfall"].unique()` returns a list of all unique values in the `"rainfall"` column of a pandas DataFrame.

```
# converting the yes & no to 1 and 0 respectively
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})
 ```
This command replaces `"yes"` with `1` and `"no"` with `0` in the `"rainfall"` column of a pandas DataFrame using the `.map()` function.

```
data.head() 
```
The command `data.head()` displays the first five rows of the pandas DataFrame `data`, helping to quickly inspect its structure and contents.

```
data.shape
```
The command `data.shape` returns a tuple `(rows, columns)`, showing the number of rows and columns in the pandas DataFrame `data`.

```
# setting plot style for all the plots
sns.set(style="whitegrid")
```
`sns.set("whitegrid")` sets a light gray grid style for Seaborn plots.

```
data.describe()
```
`data.describe()` provides summary statistics (count, mean, std, min, max, etc.) for numerical columns in a pandas DataFrame.

```
data.columns 
```
`data.columns` returns a list-like object containing the column names of the pandas DataFrame.

```
plt.figure(figsize=(15, 10))
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
  plt.subplot(3, 3, i)
  sns.histplot(data[column], kde=True)
  plt.title(f"Distribution of {column}")

plt.tight_layout()
plt.show()

```
`plt.figure(figsize=(15, 10))` sets the plot size to 15x10 inches using Matplotlib.\
This `for` loop iterates through the list of column names, assigning each column name to `column` and its index (starting from 1) to `i`. It's typically used for plotting or analyzing multiple columns in a pandas DataFrame.\
`plt.subplot(3, 3, i)` creates a grid of 3 rows and 3 columns for subplots and selects the `i`-th subplot for plotting.\
`sns.histplot(data[column], kde=True)` creates a histogram for the specified column with a Kernel Density Estimate (KDE) curve to show the data distribution.\
`plt.title(f"Distribution of {column}")` sets the title of the current subplot, dynamically displaying the column name.\
`plt.tight_layout()` adjusts subplot spacing to prevent overlapping, ensuring a clean and readable layout.\
`plt.show()` displays the plotted histograms.

```
plt.figure(figsize=(6, 4))
sns.countplot(x="rainfall", data=data)
plt.title("Distribution of Rainfall")
plt.show()
```
`plt.figure(figsize=(6, 4))` sets the plot size to 6x4 inches using Matplotlib.\
`sns.countplot(x="rainfall", data=data)` creates a bar plot showing the count of each unique value in the `"rainfall"` column using Seaborn.\
`plt.title("Distribution of Rainfall")` sets the title of the plot to "Distribution of Rainfall".\
`plt.show()` displays the plot, making it visible.

```
# correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()
```
`plt.figure(figsize=(10, 8))` sets the plot size to 10x8 inches using Matplotlib.\
`sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")` creates a heatmap showing the correlation between numerical columns in `data`, with values annotated and a "coolwarm" color scheme.\
`plt.title("Correlation heatmap")` sets the title of the heatmap plot.\
`plt.show()` displays the heatmap plot.

```
plt.figure(figsize=(15, 10))
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
  plt.subplot(3, 3, i)
  sns.boxplot(data[column])
  plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.show()
```
`plt.figure(figsize=(15, 10))` sets the plot size to 15x10 inches using Matplotlib.\
This `for` loop iterates over the list of column names, assigning each column name to `column` and its index (starting from 1) to `i`. It is commonly used for plotting or processing multiple columns in a pandas DataFrame.\
`plt.subplot(3, 3, i)` creates a grid of 3 rows and 3 columns for subplots and selects the `i`-th subplot to draw on.\
`sns.boxplot(data[column])` creates a box plot for the specified column using Seaborn to visualize its distribution and detect outliers.\
`plt.title(f"Boxplot of {column}")` sets the title of the current subplot, dynamically displaying the column name in the title.\
`plt.tight_layout()` adjusts subplot spacing to prevent overlapping, ensuring a clean and readable layout.\
`plt.show()` displays the plotted figure, making the visualizations visible.

<!-- Rishana paste your code after this comment -->

```
data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])
```

`data` Refers to a pandas DataFrame that you're working with.
   
`drop()` A pandas method used to remove specific rows or columns from a DataFrame.

`columns=['maxtemp', 'temparature', 'mintemp']`**: Specifies the names of the columns to be removed from the DataFrame. 
   - `maxtemp`: Likely the column for maximum temperature.
   - `temparature`: Appears to be a typo for `temperature`.
   - `mintemp`: Likely the column for minimum temperature.
`data =`The result of the `drop()` operation is reassigned back to `data`, so the DataFrame `data` is updated without the dropped columns.


```
data.head()
```

`data` Refers to the pandas DataFrame containing your data.
   
`head()` A pandas method used to display the first 5 rows of the DataFrame by default.

```
print(data["rainfall"].value_counts())
```
`print(data["rainfall"].value_counts())`:

`data` Refers to the pandas DataFrame containing your data.
   
`["rainfall"]` This selects the column named `"rainfall"` from the DataFrame `data`.

`value_counts()`A pandas method that counts the unique values in the specified column (`rainfall` in this case) and returns the frequency of each unique value.

`print()` Displays the output of the `value_counts()` method, which shows the count of unique values in the "rainfall" column.

```
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]
```

`df_majority = data[data["rainfall"] == 1]`**:
   - `data["rainfall"] == 1` Filters the DataFrame to select rows where the `"rainfall"` column has a value of `1`.
   - `data[...]` The filtered rows are assigned to a new DataFrame `df_majority`, which contains only the rows with `rainfall` equal to `1`.

`df_minority = data[data["rainfall"] == 0]`
   - `data["rainfall"] == 0` Filters the DataFrame to select rows where the `"rainfall"` column has a value of `0`.
   - `data[...]` The filtered rows are assigned to a new DataFrame `df_minority`, which contains only the rows with `rainfall` equal to `0`.

```
print(df_majority.shape)
print(df_minority.shape)
```

`print(df_majority.shape)`
   - `df_majority.shape` The `.shape` attribute returns the dimensions of the `df_majority` DataFrame (number of rows and columns).
   - `print()` Displays the dimensions (rows, columns) of the `df_majority` DataFrame.

`print(df_minority.shape)`
   - `df_minority.shape` The `.shape` attribute returns the dimensions of the `df_minority` DataFrame (number of rows and columns).
   - `print()` Displays the dimensions (rows, columns) of the `df_minority` DataFrame.
   
   ```
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
   ``` 

`resample()`: A function from the `sklearn.utils` module used to resample (downsample or upsample) a DataFrame or array.

`df_majority` The DataFrame containing the majority class (rows where `"rainfall"` is 1). 
`replace=False` Ensures that sampling is done without replacement, meaning no row is selected more than once.

`n_samples=len(df_minority)` Specifies that the number of rows in the downsampled `df_majority` DataFrame should match the number of rows in the `df_minority` DataFrame (to balance the classes).

`random_state=42`Sets a seed for the random number generator to ensure reproducibility of the sampling process.

`df_majority_downsampled` The result of the downsampling operation, which contains a subset of rows from the majority class (`df_majority`) with the same number of rows as the minority class.

``` 
 df_majority_downsampled.shape
```

`df_majority_downsampled` Refers to the DataFrame containing the downsampled majority class data (after using the `resample` function to match the size of the minority class).

`.shape` This attribute returns the dimensions of the DataFrame, specifically the number of rows and columns.

``` 
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
```
`pd.concat([df_majority_downsampled, df_minority])`
   -`pd.concat()` A pandas function used to concatenate two or more pandas objects (Series or DataFrames) along a particular axis (rows or columns). citeturn0search0
   - `[df_majority_downsampled, df_minority]` A list containing the two DataFrames to be concatenated. In this case, `df_majority_downsampled` (the downsampled majority class) and `df_minority` (the minority class).

`df_downsampled =`
   - Assigns the concatenated result to `df_downsampled`, creating a new DataFrame that combines the downsampled majority class and the minority class.

```
df_downsampled.shape
```
`df_downsampled` This DataFrame is created by concatenating the downsampled majority class (`df_majority_downsampled`) and the minority class (`df_minority`), resulting in a balanced dataset.

`.shape` This attribute provides the dimensions of the DataFrame:
   - The first value in the tuple indicates the total number of rows.
   - The second value indicates the total number of columns.

```
df_downsampled.head()
```
 `df_downsampled.head()`
   - `df_downsampled` Refers to the pandas DataFrame that combines the downsampled majority class and the minority class, resulting in a balanced dataset.
   - `.head()` A pandas method that returns the first 5 rows of the DataFrame by default. It's useful for quickly inspecting the initial entries of the dataset. citeturn0search0

```
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
```
 `df_downsampled.sample(frac=1, random_state=42)`
   - `sample(frac=1)` Shuffles all rows in the DataFrame. Setting `frac=1` means returning all rows in random order. citeturn0search0
   - `random_state=42` Ensures reproducibility by initializing the random number generator to a fixed state.

 `.reset_index(drop=True)`
   - `reset_index()`Resets the index of the DataFrame, assigning a new sequential index starting from 0.
   - `drop=True` Prevents the old index from being added as a new column in the DataFrame.



```python
df_downsampled.head()
``` 
`df_downsampled.head()` displays the first five rows of the pandas DataFrame `df_downsampled`, allowing for a quick inspection of its contents. 

```
df_downsampled["rainfall"].value_counts()
```
`df_downsampled["rainfall"].value_counts()` counts the occurrences of each unique value in the `"rainfall"` column and returns the results in descending order.

```
# split features and target as X and y
X = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]
```
`X = df_downsampled.drop(columns=["rainfall"])` creates a new DataFrame `X` by removing the `"rainfall"` column from `df_downsampled`, keeping only the feature variables for model training.\
`y = df_downsampled["rainfall"]` extracts the `"rainfall"` column from `df_downsampled` as the target variable for model training.

```
print(X)
```

`X` A DataFrame containing feature variables (independent variables) after removing the `"rainfall"` column.  
`print(X)` Displays the entire DataFrame `X` in the console.  

```
print(y)
```  

`y`  A Pandas Series containing the `"rainfall"` column (dependent/target variable).  
`print(y)` Displays all values in `y` in the console.  

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

`train_test_split` A function from `sklearn.model_selection` used to split data.  
`X_train, X_test` `X_train` contains training feature data, and `X_test` contains testing feature data.  
`y_train, y_test` `y_train` contains training target data, and `y_test` contains testing target data.  
`test_size=0.2` 20% of the data is used for testing, and 80% for training.  
`random_state=42` Ensures reproducibility by setting a fixed random seed.  

```
rf_model = RandomForestClassifier(random_state=42)

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
```

`rf_model = RandomForestClassifier(random_state=42)`
   - Creates a `Random Forest Classifier` instance.  
   - `random_state=42` ensures reproducibility.  

`param_grid_rf = {...}` 
   - Defines a dictionary of hyperparameters for tuning the model.  

Hyperparameters in `param_grid_rf` 
   - `n_estimators` Number of trees in the forest (**50, 100, 200**).  
   - `max_features` Number of features considered for splitting (`"sqrt"` or `"log2"`).  
   - `max_depth` Maximum depth of trees (`None` means unlimited, or set to **10, 20, 30**).  
   - `min_samples_split` Minimum samples needed to split a node (**2, 5, 10**).  
   - `min_samples_leaf` Minimum samples required at a leaf node (**1, 2, 4**).  

```
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)

grid_search_rf.fit(X_train, y_train)
```

`GridSearchCV` – A Scikit-learn function used to find the best hyperparameters for a model through exhaustive search.  

`grid_search_rf = GridSearchCV(...)`  
   - `estimator=rf_model`  Uses `rf_model` (Random Forest Classifier) as the base model.  
   - `param_grid=param_grid_rf` Tests different hyperparameter combinations from `param_grid_rf`.  
   - `cv=5` Uses `5-fold cross-validation` to evaluate each combination.  
   - `n_jobs=-1` Uses `all available CPU cores` for faster computation.  
   - `verbose=2` Prints progress updates during execution.  

`grid_search_rf.fit(X_train, y_train)`
   - Trains the model with different hyperparameter combinations.  
   - Selects the **best hyperparameter combination** based on cross-validation performance.  

```
best_rf_model = grid_search_rf.best_estimator_

print("best parameters for Random Forest:", grid_search_rf.best_params_)
```

`best_rf_model = grid_search_rf.best_estimator_`
   - Retrieves the `best model` from `grid_search_rf`.  
   - The best model is trained using the optimal hyperparameters found by `GridSearchCV`.  

`print("best parameters for Random Forest:", grid_search_rf.best_params_)` 
   - Prints the `best hyperparameter combination` found during the tuning process.  


```
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
```
Performs 5-fold cross-validation to evaluate `best_rf_model` on the training data (`X_train`, `y_train`).\
`print("Cross-validation scores:", cv_scores)` Prints the cross-validation scores to show how well the model performs across different training subsets.\
`print("Mean cross-validation score:", np.mean(cv_scores))` Prints the model's average cross-validation score.\

```
# test set performance
y_pred = best_rf_model.predict(X_test)

print("Test set Accuracy:", accuracy_score(y_test, y_pred))
print("Test set Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

`y_pred = best_rf_model.predict(X_test)` Generates predictions (`y_pred`) for the test data (`X_test`) using `best_rf_model`.\
`print("Test set Accuracy:", accuracy_score(y_test, y_pred))` Prints the accuracy of the model on the test set.\
`print("Test set Confusion Matrix:\n", confusion_matrix(y_test, y_pred))` Prints the confusion matrix to show the model's performance in classifying test data.\
`print("Classification Report:\n", classification_report(y_test, y_pred))` Prints the classification report, which includes precision, recall, and F1-score for each class.

```
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)

input_df = pd.DataFrame([input_data], columns=['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine','winddirection', 'windspeed'])
```

Stores a tuple `input_data` containing numerical values, likely representing features for a prediction model.\
Creates a Pandas DataFrame `input_df` with `input_data`, labeling columns with weather-related feature names.\

```
input_df
```
Displays the `input_df` DataFrame, showing the weather-related input data in a structured table format.

```
prediction = best_rf_model.predict(input_df)
```
Uses `best_rf_model` to predict the output for `input_df`, storing the result in `prediction`.
```
print(prediction)
```
Prints the predicted output from `best_rf_model` for the given input data.

```
prediction[0]
```
Returns the first (and likely only) predicted value from `prediction`.

```
prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")
```
Generates a prediction for `input_df` using `best_rf_model` and stores the result in `prediction`.\
Prints whether the model predicts "Rainfall" or "No Rainfall" based on the prediction result.

```
# save model and feature names to a pickle file
model_data = {"model": best_rf_model, "feature_names": X.columns.tolist()}

with open("rainfall_prediction_model.pkl", "wb") as file:
  pickle.dump(model_data, file)
  ```
Creates a dictionary `model_data` storing the trained `best_rf_model` and the feature names used in training.\
Opens a file named `rainfall_prediction_model.pkl` in write-binary (`wb`) mode to save the model.\
Saves the `model_data` dictionary (containing the trained model and feature names) to the file using Pickle.

```
import pickle
import pandas as pd
```
Imports the `pickle` module, which is used for saving and loading Python objects like machine learning models.\
Imports the Pandas library for data manipulation and analysis.

```
# load the trained model and feature names from the pickle file
with open("rainfall_prediction_model.pkl", "rb") as file:
  model_data = pickle.load(file)
```
Opens the file `rainfall_prediction_model.pkl` in read-binary (`rb`) mode to load the saved model.\
Loads the saved `model_data` dictionary from the file, retrieving the trained model and feature names.

```
model = model_data["model"]
feature_names = model_data["feature_names"]
```
Extracts the trained model from the `model_data` dictionary and stores it in the `model` variable.\
Extracts the list of feature names from `model_data` and stores it in `feature_names`.

```
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)

input_df = pd.DataFrame([input_data], columns=feature_names)
```
Stores weather-related input values as a tuple `input_data`, likely for making a prediction using the loaded model.\
Creates a Pandas DataFrame `input_df` using `input_data`, assigning column names from `feature_names` for proper formatting.

```
prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")
```
Uses `best_rf_model` to predict the outcome for `input_df` and stores the result in `prediction`. 
Prints the predicted result, displaying **"Rainfall"** if `prediction[0] == 1`, otherwise **"No Rainfall"**.



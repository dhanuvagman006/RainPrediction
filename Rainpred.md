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


```python
print(data.isnull().sum()) 
```
prints the number of missing values in each coloum of the data set.


```python
data["rainfall"].unique() 

```
retrives and display unique values in "rainfall"coloum.

### converting the yes & no to 1 and 0 respectively

```python
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})
 ```
 Converts the "rainfall" coloum from categorical values ("yes" and "No") to numerical value (1 and 0) for further anaylysis.


```python
data.head() 
```
Displays the first few rows of the datasets.


```python
data.shape
```
 prints the shape(number of rows and coloums) of the dataset.

### setting plot style for all the plots

``` python
sns.set(style="whitegrid")
```
 sets the visualization style for all plots.


```python
data.describe()
```
 provides statistical summary of the dataset,including mean,standard deviation,min,max,and quartiles.


```python
data.columns 
```
list all coloumns in a dataset


```python
plt.figure(figsize=(15, 10))
```
 creates figure with a size of 15*10inches.


  ```python
  for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
  plt.subplot(3, 3, i)
  sns.histplot(data[column], kde=True)
  plt.title(f"Distribution of {column}")
  ```
    Iterates over selected weather  parameters and plot histogram with KDE (Kernel Density Estimation) for variable.
  
 
```python
plt.tight_layout()
plt.show()
```
 Adjusts the layout and displays the histogram plots.


```python
plt.figure(figsize=(6, 4))
sns.countplot(x="rainfall", data=data)
plt.title("Distribution of Rainfall")
plt.show()
```
 creates figure with size of 6*4inches, and displays a count plot distrubution of rainfall(1=Yes,0=No).

### correlation matrix

```python
plt.figure(figsize=(10, 8)) 
```
creates  a figure with a size of 10*8 inches.


```python
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()
```
 Generates heatmap displaying correlation between weather parameters.


```python
plt.figure(figsize=(15, 10)) 
```
creates figure with size of 15*10 inches


  
  ```python
  for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
  plt.subplot(3, 3, i)
  sns.boxplot(data[column])
  plt.title(f"Boxplot of {column}") 
  ```
iterates over selected weather parameters and plots  boxplots to visualize the distribution and detect outliers.
  

```python
plt.tight_layout()
plt.show()  
```
adjusts the layout and display the boxplots.




<!-- Rishana paste your code after this comment -->

```
data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])
```

**`data`**: Refers to a pandas DataFrame that you're working with.
   
**`drop()`**: A pandas method used to remove specific rows or columns from a DataFrame.

**`columns=['maxtemp', 'temparature', 'mintemp']`**: Specifies the names of the columns to be removed from the DataFrame. 
   - `maxtemp`: Likely the column for maximum temperature.
   - `temparature`: Appears to be a typo for `temperature`.
   - `mintemp`: Likely the column for minimum temperature.

**`data =`**: The result of the `drop()` operation is reassigned back to `data`, so the DataFrame `data` is updated without the dropped columns.


```
data.head()
```

**`data`**: Refers to the pandas DataFrame containing your data.
   
**`head()`**: A pandas method used to display the first 5 rows of the DataFrame by default.

```
print(data["rainfall"].value_counts())
```
`print(data["rainfall"].value_counts())`:

**`data`**: Refers to the pandas DataFrame containing your data.
   
**`["rainfall"]`**: This selects the column named `"rainfall"` from the DataFrame `data`.

**`value_counts()`**: A pandas method that counts the unique values in the specified column (`rainfall` in this case) and returns the frequency of each unique value.

**`print()`**: Displays the output of the `value_counts()` method, which shows the count of unique values in the "rainfall" column.

```
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]
```

**`df_majority = data[data["rainfall"] == 1]`**:
   - **`data["rainfall"] == 1`**: Filters the DataFrame to select rows where the `"rainfall"` column has a value of `1`.
   - **`data[...]`**: The filtered rows are assigned to a new DataFrame `df_majority`, which contains only the rows with `rainfall` equal to `1`.

**`df_minority = data[data["rainfall"] == 0]`**:
   - **`data["rainfall"] == 0`**: Filters the DataFrame to select rows where the `"rainfall"` column has a value of `0`.
   - **`data[...]`**: The filtered rows are assigned to a new DataFrame `df_minority`, which contains only the rows with `rainfall` equal to `0`.

```
print(df_majority.shape)
print(df_minority.shape)
```

**`print(df_majority.shape)`**:
   - **`df_majority.shape`**: The `.shape` attribute returns the dimensions of the `df_majority` DataFrame (number of rows and columns).
   - **`print()`**: Displays the dimensions (rows, columns) of the `df_majority` DataFrame.

**`print(df_minority.shape)`**:
   - **`df_minority.shape`**: The `.shape` attribute returns the dimensions of the `df_minority` DataFrame (number of rows and columns).
   - **`print()`**: Displays the dimensions (rows, columns) of the `df_minority` DataFrame.
   
   ```
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
   ``` 

**`resample()`**: A function from the `sklearn.utils` module used to resample (downsample or upsample) a DataFrame or array.

**`df_majority`**: The DataFrame containing the majority class (rows where `"rainfall"` is 1). 
**`replace=False`**: Ensures that sampling is done without replacement, meaning no row is selected more than once.

**`n_samples=len(df_minority)`**: Specifies that the number of rows in the downsampled `df_majority` DataFrame should match the number of rows in the `df_minority` DataFrame (to balance the classes).

**`random_state=42`**: Sets a seed for the random number generator to ensure reproducibility of the sampling process.

**`df_majority_downsampled`**: The result of the downsampling operation, which contains a subset of rows from the majority class (`df_majority`) with the same number of rows as the minority class.

``` 
 df_majority_downsampled.shape
```

**`df_majority_downsampled`**: Refers to the DataFrame containing the downsampled majority class data (after using the `resample` function to match the size of the minority class).

**`.shape`**: This attribute returns the dimensions of the DataFrame, specifically the number of rows and columns.

``` 
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
```
**`pd.concat([df_majority_downsampled, df_minority])`**:
   - **`pd.concat()`**: A pandas function used to concatenate two or more pandas objects (Series or DataFrames) along a particular axis (rows or columns). citeturn0search0
   - **`[df_majority_downsampled, df_minority]`**: A list containing the two DataFrames to be concatenated. In this case, `df_majority_downsampled` (the downsampled majority class) and `df_minority` (the minority class).

**`df_downsampled =`**:
   - Assigns the concatenated result to `df_downsampled`, creating a new DataFrame that combines the downsampled majority class and the minority class.

```
df_downsampled.shape
```
**`df_downsampled`**: This DataFrame is created by concatenating the downsampled majority class (`df_majority_downsampled`) and the minority class (`df_minority`), resulting in a balanced dataset.

**`.shape`**: This attribute provides the dimensions of the DataFrame:
   - The first value in the tuple indicates the total number of rows.
   - The second value indicates the total number of columns.

```
df_downsampled.head()
```
 **`df_downsampled.head()`**:
   - **`df_downsampled`**: Refers to the pandas DataFrame that combines the downsampled majority class and the minority class, resulting in a balanced dataset.
   - **`.head()`**: A pandas method that returns the first 5 rows of the DataFrame by default. It's useful for quickly inspecting the initial entries of the dataset. citeturn0search0

```
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
```
 **`df_downsampled.sample(frac=1, random_state=42)`**:
   - **`sample(frac=1)`**: Shuffles all rows in the DataFrame. Setting `frac=1` means returning all rows in random order. citeturn0search0
   - **`random_state=42`**: Ensures reproducibility by initializing the random number generator to a fixed state.

 **`.reset_index(drop=True)`**:
   - **`reset_index()`**: Resets the index of the DataFrame, assigning a new sequential index starting from 0.
   - **`drop=True`**: Prevents the old index from being added as a new column in the DataFrame.

<!-- Rakshitha paste your code after this comment -->
Start from here Rish
```
df_downsampled.head()
```
**`df_downsampled`** – This is a Pandas DataFrame that has likely been downsampled (i.e., reduced in size by sampling or aggregation).
**`.head()`** – This function returns the first five rows of the DataFrame by default.

```
df_downsampled["rainfall"].value_counts()
```

**`df_downsampled`** – A Pandas DataFrame that has been downsampled.  
**`["rainfall"]`** – Selects the "rainfall" column from the DataFrame.  
**`.value_counts()`** – Counts the occurrences of each unique value in the "rainfall" column.  

```
X = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]
```

**`df_downsampled`** – A Pandas DataFrame that contains the dataset.  
**`drop(columns=["rainfall"])`** – Removes the "rainfall" column from `df_downsampled`, keeping only feature columns.  
**`X = ...`** – Stores the remaining feature columns in `X` (independent variables).  
**`y = df_downsampled["rainfall"]`** – Stores the "rainfall" column as `y` (dependent/target variable).  

```
print(X)
```

**`X`** – A DataFrame containing feature variables (independent variables) after removing the `"rainfall"` column.  
**`print(X)`** – Displays the entire DataFrame `X` in the console.  

```
print(y)
```  

**`y`** – A Pandas Series containing the `"rainfall"` column (dependent/target variable).  
**`print(y)`** – Displays all values in `y` in the console.  

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


**`train_test_split`** – A function from `sklearn.model_selection` used to split data.  
**`X_train, X_test`** – `X_train` contains training feature data, and `X_test` contains testing feature data.  
**`y_train, y_test`** – `y_train` contains training target data, and `y_test` contains testing target data.  
**`test_size=0.2`** – 20% of the data is used for testing, and 80% for training.  
**`random_state=42`** – Ensures reproducibility by setting a fixed random seed.  

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

**`rf_model = RandomForestClassifier(random_state=42)`**  
   - Creates a **Random Forest Classifier** instance.  
   - `random_state=42` ensures reproducibility.  

**`param_grid_rf = {...}`**  
   - Defines a dictionary of hyperparameters for tuning the model.  

**Hyperparameters in `param_grid_rf`**:  
   - **`n_estimators`**: Number of trees in the forest (**50, 100, 200**).  
   - **`max_features`**: Number of features considered for splitting (`"sqrt"` or `"log2"`).  
   - **`max_depth`**: Maximum depth of trees (`None` means unlimited, or set to **10, 20, 30**).  
   - **`min_samples_split`**: Minimum samples needed to split a node (**2, 5, 10**).  
   - **`min_samples_leaf`**: Minimum samples required at a leaf node (**1, 2, 4**).  

```
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)

grid_search_rf.fit(X_train, y_train)
```

**`GridSearchCV`** – A Scikit-learn function used to find the best hyperparameters for a model through exhaustive search.  

**`grid_search_rf = GridSearchCV(...)`**  
   - **`estimator=rf_model`** → Uses `rf_model` (Random Forest Classifier) as the base model.  
   - **`param_grid=param_grid_rf`** → Tests different hyperparameter combinations from `param_grid_rf`.  
   - **`cv=5`** → Uses **5-fold cross-validation** to evaluate each combination.  
   - **`n_jobs=-1`** → Uses **all available CPU cores** for faster computation.  
   - **`verbose=2`** → Prints progress updates during execution.  

**`grid_search_rf.fit(X_train, y_train)`**  
   - Trains the model with different hyperparameter combinations.  
   - Selects the **best hyperparameter combination** based on cross-validation performance.  

```
best_rf_model = grid_search_rf.best_estimator_

print("best parameters for Random Forest:", grid_search_rf.best_params_)
```

**`best_rf_model = grid_search_rf.best_estimator_`**  
   - Retrieves the **best model** from `grid_search_rf`.  
   - The best model is trained using the optimal hyperparameters found by **GridSearchCV**.  

**`print("best parameters for Random Forest:", grid_search_rf.best_params_)`**  
   - Prints the **best hyperparameter combination** found during the tuning process.  

```
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores)) 
``` 

**`cross_val_score(best_rf_model, X_train, y_train, cv=5)`**  
   - Evaluates `best_rf_model` using **5-fold cross-validation**.  
   - Splits `X_train, y_train` into 5 subsets (folds) and trains the model on 4 folds while testing on the remaining fold, repeating for all 5.  
   - Returns an array of accuracy scores (one for each fold).  

**`cv_scores = ...`**  
   - Stores the cross-validation scores as an array.  

**`print("Cross-validation scores:", cv_scores)`**  
   - Prints the individual cross-validation accuracy scores.  

**`np.mean(cv_scores)`**  
   - Computes the **mean cross-validation score** (average model performance across all folds).  


<!-- line 44 to 57 code after this comment -->

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

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
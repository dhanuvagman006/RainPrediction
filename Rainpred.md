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


<!-- Rishana paste your code after this comment -->


```
data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])
```

1. **`data`**: Refers to a pandas DataFrame that you're working with.
   
2. **`drop()`**: A pandas method used to remove specific rows or columns from a DataFrame.

3. **`columns=['maxtemp', 'temparature', 'mintemp']`**: Specifies the names of the columns to be removed from the DataFrame. 
   - `maxtemp`: Likely the column for maximum temperature.
   - `temparature`: Appears to be a typo for `temperature`.
   - `mintemp`: Likely the column for minimum temperature.

4. **`data =`**: The result of the `drop()` operation is reassigned back to `data`, so the DataFrame `data` is updated without the dropped columns.


```
data.head()
```

1. **`data`**: Refers to the pandas DataFrame containing your data.
   
2. **`head()`**: A pandas method used to display the first 5 rows of the DataFrame by default.

```
print(data["rainfall"].value_counts())
```
Here’s a short, point-wise explanation of the code `print(data["rainfall"].value_counts())`:

1. **`data`**: Refers to the pandas DataFrame containing your data.
   
2. **`["rainfall"]`**: This selects the column named `"rainfall"` from the DataFrame `data`.

3. **`value_counts()`**: A pandas method that counts the unique values in the specified column (`rainfall` in this case) and returns the frequency of each unique value.

4. **`print()`**: Displays the output of the `value_counts()` method, which shows the count of unique values in the "rainfall" column.

```
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]
```

1. **`df_majority = data[data["rainfall"] == 1]`**:
   - **`data["rainfall"] == 1`**: Filters the DataFrame to select rows where the `"rainfall"` column has a value of `1`.
   - **`data[...]`**: The filtered rows are assigned to a new DataFrame `df_majority`, which contains only the rows with `rainfall` equal to `1`.

2. **`df_minority = data[data["rainfall"] == 0]`**:
   - **`data["rainfall"] == 0`**: Filters the DataFrame to select rows where the `"rainfall"` column has a value of `0`.
   - **`data[...]`**: The filtered rows are assigned to a new DataFrame `df_minority`, which contains only the rows with `rainfall` equal to `0`.

# Rain Prediction
## Overview
This project aims to predict rainfall using machine learning techniques. It leverages Python libraries for data processing, visualization, and model training.

## Drpendencies
Ensure you have the following Python libraries installed:

-  `numpy` -Numerical computations
- `pandas` -Data manipulation and analysis
- `matplotlib` -Data visualization
- `seaborn` -Statistical data visualization
- `scikit-Learn` -Machine learning tools
- `pickle` -Model saving and loading
### Installation
You can install them using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
## I mporting Libraries
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
## checking the number of missing values


```
`print(data.isnull().sum())`prints the ** number of missing values** in each coloum of the data set.
```

```
`data["rainfall"].unique()` retrives and display unique values in "rainfall"coloum.
```
## converting the yes & no to 1 and 0 respectively

```
`data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})` Converts the "rainfall" coloum from categorical values ("yes" and "No") to numerical value (1 and 0) for further anaylysis.
```

```
`data.head()` Displays the first few rows of the datasets.
```

```
`data.shape` prints the shape(number of rows and coloums) of the dataset.
```
## setting plot style for all the plots

``` 
`sns.set(style="whitegrid")` sets the visualization style for all plots.
```

```
`data.describe()` provides statistical summary of the dataset,including mean,standard deviation,min,max,and quartiles.
```

```
`data.columns` list all coloumns in a dataset
```

```
`plt.figure(figsize=(15, 10))` creates figure with a size of 15*10inches.
```

  ```
  `for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
  plt.subplot(3, 3, i)
  sns.histplot(data[column], kde=True)
  plt.title(f"Distribution of {column}")` I terates over selected weather  parameters and plot histogram with KDE (Kernel Density Estimation) for variable.
  ```
 
```
`plt.tight_layout()
plt.show()`Adjusts the layout and displays the histogram plots.
```

```
`plt.figure(figsize=(6, 4))
sns.countplot(x="rainfall", data=data)
plt.title("Distribution of Rainfall")
plt.show()`creates figure with size of 6*4inches, and displays a count plot distrubution of rainfall(1=Yes,0=No).
```
## correlation matrix

```
`plt.figure(figsize=(10, 8))`creates  a figure with a size of 10*8 inches.
```

```
`sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()` Generates heatmap displaying correlation between weather parameters.
```

```
`plt.figure(figsize=(15, 10))`creates figure with size of 15*10 inches
```

  
  ```
 ` for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
  plt.subplot(3, 3, i)
  sns.boxplot(data[column])
  plt.title(f"Boxplot of {column}")`iterates over selected weather parameters and plots  boxplots to visualize the distribution and detect outliers.
  ```

```
`plt.tight_layout()
plt.show()`adjusts the layout and display the boxplots.
```


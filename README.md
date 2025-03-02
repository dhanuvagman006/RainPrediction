# Rain Prediction

## Overview
This project predicts rainfall using machine learning techniques. It leverages Python libraries for data processing, visualization, and model training.

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

## Dataset Loading
Load the dataset into a Pandas DataFrame:
```python
data = pd.read_csv("/content/Rainfall.csv")
data.head()
```

## Data Cleaning and Preprocessing
### Remove Extra Spaces in Column Names
```python
data.columns = data.columns.str.strip()
```
### Check Column Names
```python
data.columns
```
### Display Dataset Information
```python
print("Data Info:")
data.info()
```
### Drop Unnecessary Columns
```python
data = data.drop(columns=["day"])
data.head()
```
### Handle Missing Values
```python
print(data.isnull().sum())
```
Replace missing values with appropriate statistics:
```python
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())
```
### Check Unique Values
```python
data["winddirection"].unique()
```



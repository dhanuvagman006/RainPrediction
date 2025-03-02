#RainPrediction
```
import numpy as np #loads NumPy for fast math and array operations.
import pandas as pd #loads Pandas for easy data handling and analysis.
import matplotlib.pyplot as plt #loads Matplotlib for creating charts and graphs.
import seaborn as sns #loads Seaborn for advanced and beautiful data visualizations.
from sklearn.utils import resample #imports a function to randomly resample data, useful for bootstrapping and balancing datasets.
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score # imports functions for splitting data, tuning hyperparameters, and evaluating models.
from sklearn.ensemble import RandomForestClassifier #Imports a decision-tree-based classifier.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #Imports functions to evaluate model performance.
import pickle #Imports Pickle for saving and loading Python objects.
data = pd.read_csv("/content/Rainfall.csv") #

```



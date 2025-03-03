

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



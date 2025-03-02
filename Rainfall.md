# Checking the number of missing values in the DataFrame
print(data.isnull().sum())  # Corrected to call sum() to get the actual counts of missing values

# Displaying the unique values in the 'rainfall' column to understand its categories
data["rainfall"].unique()

# Converting the 'rainfall' column values from 'yes' and 'no' to 1 and 0, respectively
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})

# Displaying the first five rows of the DataFrame to inspect the changes made
data.head()

# Getting the shape of the DataFrame, which returns the number of rows and columns
data.shape

# Setting the plot style for all plots to 'whitegrid' for better readability
sns.set(style="whitegrid")

# Generating descriptive statistics for the DataFrame, including count, mean, std, min, max, and quartiles
data.describe()

# Displaying the names of the columns in the DataFrame to understand the available features
data.columns

# Creating a new figure for plotting with a specified size of 15 inches wide and 10 inches tall
plt.figure(figsize=(15, 10))

# Looping through a list of specified column names to create histograms for each
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed'], 1):
    # Creating a subplot in a 3x3 grid for the current iteration
    plt.subplot(3, 3, i)
    
    # Creating a histogram for the current column with a KDE overlay to visualize its distribution
    sns.histplot(data[column], kde=True)
    
    # Setting the title of the current subplot to indicate which column's distribution is being displayed
    plt.title(f"Distribution of {column}")

# Adjusting the spacing between subplots to prevent overlap and ensure a clean layout
plt.tight_layout()

# Displaying all the plots created so far
plt.show()

# Creating a new figure for plotting the count of rainfall occurrences with a specified size of 6 inches wide and 4 inches tall
plt.figure(figsize=(6, 4))

# Creating a count plot for the 'rainfall' column to show the number of occurrences of each category (0 and 1)
sns.countplot(x="rainfall", data=data)

# Setting the title of the count plot to indicate that it shows the distribution of rainfall
plt.title("Distribution of Rainfall")

# Displaying the count plot created for the rainfall distribution
plt.show()

# Creating a new figure for plotting the correlation matrix with a specified size of 10 inches wide and 8 inches tall
plt.figure(figsize=(10, 8))

# Creating a heatmap of the correlation matrix of the DataFrame, with annotations and a color map
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")

# Setting the title of the heatmap to indicate that it shows the correlation between features
plt.title("Correlation heatmap")

# Displaying the heatmap created for the correlation matrix
plt.show()

# Creating a new figure for plotting boxplots with a specified size of 15 inches wide and 10 inches tall
plt.figure(figsize=(15, 10))

# Looping through a list of specified column names to create boxplots for each
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed'], 1):
    # Creating a subplot in a 3x3 grid for the current iteration
    plt.subplot(3, 3, i)
    
    # Creating a boxplot for the current column to visualize its distribution and identify outliers
    sns.boxplot(data[column])
    
    # Setting the title of the current subplot to indicate which column's boxplot is being displayed
    plt.title(f"Boxplot of {column}")

# Adjusting the spacing between subplots to prevent overlap and ensure a clean layout
plt.tight_layout()

# Displaying all the boxplots created so far
plt.show()'
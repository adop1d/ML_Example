# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)

#print rows
print(data.head())

#print summary

data.info()

data = data.dropna()

# Select features and target variable
features = ['displacement', 'horsepower', 'weight', 'acceleration', 'cylinders']
target = 'mpg'

correlation_matrix = data[features + [target]].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Feature Importance')
plt.show()


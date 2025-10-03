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

#Selecting most relevant features based on correlation
X=data[['displacement', 'horsepower', 'weight', 'cylinders']].values
Y=data['mpg'].values

#Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error: {mse}.2f")

#plot the true vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, y_pred, alpha=0.75, color='b',label='Predicted vs Actual')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='r', linestyle='--', label='Ideal Prediction Line')

#Calculate and plot prediction errors

for (true_value, predicted_value) in zip(Y_test, y_pred):
    plt.plot([true_value, true_value], [true_value, predicted_value], color='g', linestyle='-', linewidth=0.5)


plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs Predicted MPG with Prediction Errors')
plt.legend()
plt.grid(True)
plt.show()
data.info()
print(data.describe())

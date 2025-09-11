import pandas as pd
import certifi
import ssl
import numpy as np
from urllib.request import urlopen
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Load the Iris dataset from a URL
ssl_context = ssl.create_default_context(cafile=certifi.where())
with urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            context=ssl_context) as response:
      iris_data = pd.read_csv(response, header=None)
# Assign column names to the dataset

iris_data.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
y = iris_data['Species']
X = iris_data.drop(columns=['Species'])
#split the data into features(x) and labels(y)
print(X.head())

model = LogisticRegression()
model.fit(X.values, y)

predictions = model.predict([[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5], [7.2, 3.6, 6.1, 2.5]])
print(predictions)




import numpy as np
import pandas as pd
from kNearestNeighbor import kNearestNeighbors

data = pd.read_csv('Social_Network_Ads.csv')

# Debug Point 1
# print(data.head())

X = data.iloc[:, 2:4].values
y = data.iloc[:, -1].values

# Debug Point 2
# print(X.shape)
# print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Debug Point 3
# print(X_train.shape)
# print(X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Debug Point 4
# print(X_train)
# print(X_test)

# Object of kNN
knn = kNearestNeighbors(k = 5)

knn.fit(X_train, y_train)

# print(X_train[0])

# knn.predict(np.array([60, 100000]).reshape(1, 2))

def predict_new():
    age = int(input("Enter the age: "))
    salary = int(input('Enter the salary: '))
    X_new = np.array([[age], [salary]]).reshape(1, 2)

    X_new = scaler.transform(X_new)

    result = knn.predict(X_new)

    print("Will not purchase") if result == 0 else print("Will purchase")

predict_new()


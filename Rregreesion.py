import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
'''
# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
'''
# Create linear regression object
regr = linear_model.LinearRegression()
X_test = np.array([1690,847,1909,1068,46.77,1920,1080])
X_test = X_test.reshape(1, -1)
X_train = np.array([[1720,880,1910,1069,0,1920,1080],[1655,814,1907,1066,100,1920,1080]])
y_train = np.array([[1729,853,1766,865,49],[1668,778,1716,795,65]])
# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
print(int(y_pred))

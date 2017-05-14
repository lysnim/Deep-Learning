import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[["X"]]
y_values = dataframe[["Y"]]

# Train model on data

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)
print r2_score(y_values, body_reg.predict(y_values))

# Visualize results

plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()

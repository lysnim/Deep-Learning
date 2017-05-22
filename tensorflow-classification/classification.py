import pandas as pd                 # A beautiful library to help us work with data as tables
import numpy as np                  # So we can use number matrices. Both pandas and TensorFlow need it.
import matplotlib.pyplot as plt     # Visualize the things
import tensorflow as tf             # Fire from the gods

dataframe = pd.read_csv("data.csv") # Let's have Pandas load our dataset as a dataframe
dataframe = dataframe.drop(["index", "price", "sq_price"], axis=1)  # Remove columns we don't care about
dataframe = dataframe[0:10]         # We'll only use the first 10 rows of the dataset in this example
print dataframe                           # Let's have the notebook show us how the dataframe looks now

dataframe.loc[:, ("y1")] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
dataframe.loc[:, ("y2")] = dataframe["y1"] == 0
dataframe.loc[:, ("y2")] = dataframe["y2"].astype(int)
print dataframe

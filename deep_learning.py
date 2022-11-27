import numpy as np
from sklearn.metrics import mean_squared_error
import json
from data_processing import create_new_features
from data_processing import train_data
import matplotlib.pyplot as plt

def train_nn(X, y):


def predict_nn(reg, X, y):
    preds = reg.predict(X)
    model_error = mean_squared_error(y, preds)
    print("The mean squared error of the optimal model is model_error:{}".format(model_error))
    return preds 

if __name__ == "__main__":
    # Train NN on training data 
    data = create_new_features(train_data)
    X = data[:, [0, 1]]
    y = data[:, 2]
    reg = train_nn(X, y)
    preds = predict_nn(reg, X, y)
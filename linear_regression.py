from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import json
from raw_data import *
from data_preprocessing import *
import matplotlib.pyplot as plt

def train_linear_regression(X, y):
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    coeffs = reg.coef_
    intercept = reg.intercept_

    # Save coefficients and scores as a dictionary in file
    regression_output = {"score": score, "coeffs": coeffs.tolist(), "intercept": intercept}
    print(regression_output)

    # create json object from dictionary
    json_obj = json.dumps(regression_output)

    # open file for writing, "w" 
    f = open("regression_output.json","w")

    # write json object to file
    f.write(json_obj)

    # close file
    f.close()

    return reg

def predict_linear_regression(reg, X, y):
    preds = reg.predict(X)
    model_error = mean_squared_error(y, preds)
    print("The mean squared error of the optimal model is model_error:{}".format(model_error))
    return preds 

"""
  Create a scatter plot of X and y data, with y_pred determining a line.
  Input: 
    X - x value vector
    y - y value vector
    y_pred - prediction values vector
  Output:
    None
"""
def create_plot(X, y, y_pred):
  plt.scatter(X, y, color="black")
  plt.plot(X, y_pred, color="blue", linewidth=2)
  plt.title("BT-easiness versus CEFR score")

  plt.xticks(())
  plt.yticks(())

  plt.show()

"""
  Run the regression on train data for the CEFR ratings. One baseline.
"""
def cefr_baseline():
  data = cefr_train()
  X = data[:, [0, 1]]
  y = data[:, 2]
  reg = train_linear_regression(X, y)
  preds = predict_linear_regression(reg, X, y)

  return preds

"""
  Run the regression on train data for average word length and average
  sentence length. One baseline.
"""
def sentence_word_len_baseline():
  data = create_new_features()
  X = data[:, [0, 1]]
  y = data[:, 2]
  reg = train_linear_regression(X, y)
  preds = predict_linear_regression(reg, X, y)

  return preds

"""
  Run regression with the bag-of-words representation.
"""
def bag_of_words_regression():
  X_data = text_pre_processing()
  X = X_data
  y = bt_easiness_train_data()
  reg = train_linear_regression(X, y)
  preds = predict_linear_regression(reg, X, y)

  return preds

if __name__ == "__main__":
    print(bag_of_words_regression())

    # Create a plot of CEFR data against BT_easiness with the regression line
    # create_plot(X, y, preds)
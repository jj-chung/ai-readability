from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import json
from raw_data import *
from data_preprocessing import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def train_linear_regression(X, y):
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    coeffs = reg.coef_
    intercept = reg.intercept_

    # Save coefficients and scores as a dictionary in file
    regression_output = {"score": score, "coeffs": coeffs.tolist(), "intercept": intercept}

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

    return [preds, model_error] 

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
  return regression(X, y)

"""
  Run the regression on train data for average word length and average
  sentence length (baseline) and other features (non-baseline).
"""
def sentence_word_len(baseline):
  data = create_new_features("train", baseline)
  X = data[:, :-1]
  y = data[:, -1]
  return regression(X, y)

"""
  Run regression with the bag-of-words representation.
"""
def bag_of_words_regression():
  X = word_vectorizer_train()
  y = bt_easiness_train_data()
  return regression(X, y)

def regression(X, y):
  train_errs = []
  val_errs = []
  kf = KFold(n_splits = 5)

  for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    reg = train_linear_regression(X_train, y_train)
    train_preds, train_err = predict_linear_regression(reg, X_train, y_train)
    val_preds, val_err = predict_linear_regression(reg, X_val, y_val)

    train_errs.append(train_err)
    val_errs.append(val_err)
  
  return "Avg train err: {}, Avg val err: {}".format(np.average(train_errs), np.average(val_errs))

if __name__ == "__main__":
    print(sentence_word_len(True))
    print(sentence_word_len(False))

    # Create a plot of CEFR data against BT_easiness with the regression line
    # create_plot(X, y, preds)
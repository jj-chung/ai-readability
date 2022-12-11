from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import json
from raw_data import *
from data_preprocessing import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import StrMethodFormatter

def train_regression(X, y, ridge=False):
  clf = LinearRegression()
  alpha_range = range(2800, 3000, 10)

  if ridge:
    parameters = {'alpha': alpha_range}
    ridge = Ridge()
    clf = GridSearchCV(ridge, parameters)

    clf = clf.fit(X, y)
    reg = clf.best_estimator_
    score = reg.score(X, y)
    coeffs = reg.coef_
    intercept = reg.intercept_

  clf = clf.fit(X, y)

  # Create and save plot for gridsearch
  if ridge: 
    scores = clf.cv_results_['mean_test_score']
    scores = np.array(scores).reshape(len(alpha_range))
    print(scores)

    plt.plot(alpha_range, scores)

    plt.xlabel('Alpha')
    plt.ylabel('Mean score')
    plt.savefig('images/gridsearch_ridge_regression.png')

    # Save coefficients and scores as a dictionary in file
    regression_output = {"score": score, "coeffs": coeffs.tolist(), "intercept": intercept,
      "best_params": clf.best_params_}

    print(regression_output)

    # create json object from dictionary
    # json_obj = json.dumps(regression_output)

    # open file for writing, "w" 
    # f = open("regression_output.json","w")

    # write json object to file
    # f.write(json_obj)

    # close file
    # f.close()

    return reg

  return clf

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
  plt.scatter(X, y, color="blue", s = 2)
  plt.plot(X, y_pred, color="red", linewidth=2)
  plt.rcParams.update({'font.size': 20})
  plt.title("BT-easiness versus CEFR score")

  plt.xticks(np.arange(0.0, 6.0, step=1.0))
  plt.yticks(np.arange(-4.0, 3.0, step=1.0))
  plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # No decimal places
  plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  plt.savefig('images/cefr_baseline_plot.png')

"""
  Run the regression on train data for the CEFR ratings. One baseline.
"""
def cefr_baseline(train_X, train_y, test_X, test_y):
  results = {}

  clf = train_regression(train_X, train_y)
  results["train"] = predict_linear_regression(clf, train_X, train_y)
  results["test"] = predict_linear_regression(clf, test_X, test_y)

  return results

"""
  Run the regression on train data for average word length and average
  sentence length (baseline) and other features (non-baseline).
"""
def sentence_word_len(baseline):
  train_data = create_new_features("train", baseline=baseline, preprocessed=False)
  X_train = train_data[:, :-1]
  y_train = train_data[:, -1]
  
  test_data = create_new_features("test", baseline=baseline, preprocessed=False)
  X_test = test_data[:, :-1]
  y_test = test_data[:, -1]
  clf = None

  if not baseline:
    clf = train_regression(X_train, y_train, ridge=True)
  else:
    clf = train_regression(X_train, y_train, ridge=False)

  results = {}
  results["train"] = predict_linear_regression(clf, X_train, y_train)
  results["test"] = predict_linear_regression(clf, X_test, y_test)

  return results

"""
  Run regression with the bag-of-words representation.
"""
def bag_of_words_regression():
  X = word_vectorizer("train")
  y = bt_easiness_train_data()
  return regression(X, y)

def regression(X, y, ridge=False, kfold=True):
  if not kfold:
    reg = train_regression(X, y, ridge)
    return reg

  kf = KFold(n_splits = 5)
  train_errs = []
  val_errs = []

  for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    reg = train_regression(X_train, y_train, ridge)
    train_preds, train_err = predict_linear_regression(reg, X_train, y_train)
    val_preds, val_err = predict_linear_regression(reg, X_val, y_val)

    train_errs.append(train_err)
    val_errs.append(val_err)
  
  return "Avg train err: {}, Avg val err: {}".format(np.average(train_errs), np.average(val_errs))

if __name__ == "__main__":
    train_X, train_y = CEFR("train")
    test_X, test_y = CEFR("test")
    # results_cefr = cefr_baseline(train_X, train_y, test_X, test_y)
    # results_baseline = sentence_word_len(baseline=True)
    results_regression_all = sentence_word_len(baseline=False)

    # print(f'CEFR: {results_cefr}')
    # print(f'Regression: {results_baseline}')
    print(f'Baseline Regression: {results_regression_all}, {output}')

    # Create a plot of CEFR data against BT_easiness with the regression line for baseline
    # test_preds = results_cefr["test"][0]
    # create_plot(test_X.tolist(), test_y.tolist(), test_preds.tolist())
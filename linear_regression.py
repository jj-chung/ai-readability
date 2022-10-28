from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import json

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

if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    reg = train_linear_regression(X, y)
    predict_linear_regression(reg, X, y)
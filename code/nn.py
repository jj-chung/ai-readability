import numpy as np
from sklearn.metrics import mean_squared_error
import json
import data_preprocessing
from raw_data import *
from data_preprocessing import *
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

def train_nn(train_v, train_s, test_v, test_s):
    #model = MLPRegressor(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
    #mlp.fit(train_v, train_s)
    print(MLPRegressor().get_params().keys())
    parameters = {'batch_size': ['''20''', 32, 64, 128],
                    'max_iter': [150, 250, '''400'''],
                  'solver': ['sgd'], #'lbfgs', adam
                  'hidden_layer_sizes': [(1000), (1000, 1000)]}
    grid_search = GridSearchCV(estimator=MLPRegressor(),
                               param_grid=parameters,
                               scoring='neg_mean_absolute_error',
                               cv=5)
    model = grid_search.fit(train_v, train_s)

    print('Best batch_size:', grid_search.best_estimator_.get_params()['batch_size'])
    print('Best max_iter:', grid_search.best_estimator_.get_params()['max_iter'])
    print('Best solver:', grid_search.best_estimator_.get_params()['solver'])
    print('Best hidden_layer_sizes:', grid_search.best_estimator_.get_params()['hidden_layer_sizes'])

    predict_test = model.predict(test_v)
    error = mean_squared_error(test_s, predict_test)
    return error

'''def create_plot(X, y, y_pred):
    plt.scatter(X, y, color="black")
    plt.plot(X, y_pred, color="blue", linewidth=2)
    plt.title("BT-easiness versus CEFR score")

    plt.xticks(())
    plt.yticks(())

    plt.show()'''


if __name__ == "__main__":
    #---Using TDIF Vectorization---
    train_vector = data_preprocessing.word_vectorizer_train(type="train")
    train_score = bt_easiness_train_data()
    test_vector = data_preprocessing.word_vectorizer_train(type="test")
    test_score = bt_easiness_test_data()
    accuracy = train_nn(train_vector, train_score, test_vector, test_score)
    print('nn accuracy:', accuracy)


import numpy as np
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
import json
from raw_data import *
from data_preprocessing import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.model_selection import KFold
import data_preprocessing
import tensorflow as tf
import scipy as sp

#stole this shit
def neural_network(X_train, y_train, Validation_data, metrics=['mean_squared_error', 'mean_absolute_error'],
                   activation='relu', input_shape=(None, 3), optimizer='adam', loss='mean_squared_error',
                   epochs=10, batch_size=64, verbose=1):
    """
    We are defining a neural network function that takes into account a different set of parameters
    that are needed to build the machine learning model and we are also giving different values
    and it would be working with different parameters and we are able to give those values to our
    deep learning models and we are going to return the output given by the model respectively.
    """

    model = Sequential()
    model.add(Dense(500, activation=activation, input_shape=input_shape))
    model.add(Dense(100, activation=activation))
    model.add(Dense(50, activation=activation))
    model.add(Dense(10, activation=activation))
    model.add(Dense(5, activation=activation))
    model.add(Dense(1))
    model.summary()
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    if Validation_data:
        model.fit(x=X_train, y=y_train, validation_data=Validation_data, epochs=epochs, batch_size=batch_size,
                  verbose=verbose)
    else:
        '''x1 = np.asarray(X_train).astype('float32')
        y1 = np.asarray(y_train).astype('float32')'''
        model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def train_nn(X, y):
  model = Sequential()
  model.add(Dense(500, input_dim=2000, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(5, activation='relu'))
  model.add(Dense(1, activation='relu'))
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
  model.summary()
  model.fit(X, epochs=200, batch_size=15)

  # implement this once we have validation set
  '''estimator = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=100)
  train_errs = []
  val_errs = []
  kf = KFold(n_splits=10, random_state=1)

  for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    reg = estimator.fit(X_train, y_train)
    train_preds, train_err = estimator.predict(reg, X_train, y_train)
    val_preds, val_err = estimator.predict(reg, X_val, y_val)

    train_errs.append(train_err)
    val_errs.append(val_err)

  return "Avg train err: {}, Avg val err: {}".format(np.average(train_errs), np.average(val_errs))'''
  return model

def predict_nn(model, test_vect, test_scores):
  _, accuracy = model.evaluate(test_vect, test_scores, verbose=0)
  return accuracy

if __name__ == "__main__":
    # Train NN on training data 
    train_vector = data_preprocessing.create_new_features(type="train")
    #tensor1 = train_vector.astype('float32')
    #tensor1 = np.asarray(train_vector).astype('float32')
    train_vector_tensor = train_vector.astype('float32')

    train_score = bt_easiness_train_data()
    #y_train_tesor = np.asarray(train_score).astype('float32')
    train_score_tensor = train_score.astype('float32')
    test_vector = data_preprocessing.create_new_features(type="test")
    #tensor2 = test_vector.astype('float32')
    #tensor2 = np.asarray(test_vector).astype('float32')
    test_vector_tensor = test_vector.astype('float32')

    test_score = bt_easiness_test_data()
    #test_score_tensor = np.asarray(test_score).astype('float32')
    test_score_tensor = test_score.astype('float32')
    '''model = train_nn(train_vector, train_score)
    accuracy = predict_nn(model, test_vector, test_score)
    print('nn accuracy:', accuracy)'''

    model1 = neural_network(X_train=train_vector_tensor, y_train=train_score_tensor, Validation_data=None, batch_size=64)
    predicted = model1.predict(test_vector_tensor)
    print(mean_squared_error(predicted, test_score_tensor))
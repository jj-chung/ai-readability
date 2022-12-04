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
import json
import imbalanced

"""
Train a neural network using keras.
"""
def neural_network(X_train, y_train, Validation_data, metrics=['mean_squared_error', 'mean_absolute_error'],
                   activation='relu', input_shape=(None, 3), optimizer='adam', loss='mean_squared_error',
                   epochs=10, batch_size=64, verbose=1):
    model = Sequential()
    model.add(Dense(500, activation=activation, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.summary()
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    if Validation_data:
        model.fit(x=X_train, y=y_train, validation_data=Validation_data, epochs=epochs, batch_size=batch_size,
                  verbose=verbose)
    else:
        model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

def nn_train(type="train"):
  # Train NN on training data 
  train_vector = data_preprocessing.create_new_features(type=type)
  train_vector = train_vector.astype('float32')
  train_bt_easiness = bt_easiness_train_data()
  train_bt_easiness = train_bt_easiness.astype('float32')
  
  fold_num = 0
  kf = KFold(n_splits=5, shuffle=True)

  # Do k-fold validation here
  for train_index, test_index in kf.split(train_vector):
    print(f'Training on fold {fold_num}')
    X_train, X_test = train_vector[train_index], train_vector[test_index]
    y_train, y_test = test_vector[train_index], test_vector[test_index]

    #Resample the training data
    # X_train, y_train = imbalanced.resample(X_train, y_train, sample_type=sample_type)

    # Train the model on the train data
    model = neural_network(X_train, y_train, Validation_data=None, batch_size=64)

    nn_predict(model, X_test, y_test, fold_num)

    fold_num += 1
  
  return model

"""
Make predictions with a trained neural net model.

Input:
  model - NN model.
  test_vector - test vector input.
  k_val - the fold number if using k fold validation.
"""
def nn_predict(model, test_vector, test_bt_easiness, k_val=0):
  predicted = model.predict(test_vector)
  MSE = mean_squared_error(predicted, test_bt_easiness)
  _, accuracy = model.evaluate(test_vector, test_bt_easiness, verbose=0)

  results = {
    "MSE": MSE,
    "predicted_labels": predicted,
    "accuracy": accuracy
  }

  # Save predictions and mean square error to json
  with open(f'keras_data/MSE_and_predictions_{k_val}.json', 'w') as fp:
      json.dump(results, fp)

  return results

if __name__ == "__main__":
  
  # Testing data
  test_vector = data_preprocessing.create_new_features(type="test")
  test_vector = test_vector.astype('float32')
  test_bt_easiness = bt_easiness_test_data()
  test_bt_easiness = test_bt_easiness.astype('float32')
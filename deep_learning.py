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

def predict_nn(model, test_vect, test_scores):
  _, accuracy = model.evaluate(test_vect, test_scores, verbose=0)
  return accuracy

if __name__ == "__main__":
    # Train NN on training data 
    train_vector = data_preprocessing.create_new_features(type="train")
    train_vector_tensor = train_vector.astype('float32')
    train_score = bt_easiness_train_data()
    train_score_tensor = train_score.astype('float32')

    # Testing data
    test_vector = data_preprocessing.create_new_features(type="test")
    test_vector_tensor = test_vector.astype('float32')
    test_score = bt_easiness_test_data()
    test_score_tensor = test_score.astype('float32')

    model = neural_network(X_train=train_vector_tensor, y_train=train_score_tensor, Validation_data=None, batch_size=64)
    predicted = model.predict(test_vector_tensor)

    # Save mean square error to data file
    print(mean_squared_error(predicted, test_score_tensor))
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


def train_nn(X, y):
  # Build NN model, trying two layers with 1000 neurons
  model = Sequential()
  model.add(Dense(1000, input_dim=2000, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(1000, activation='relu'))
  model.add(Dropout(0.3))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    train_vector = data_preprocessing.word_vectorizer_train(type="train")
    train_score = bt_easiness_train_data()
    test_vector = data_preprocessing.word_vectorizer_train(type="test")
    test_score = bt_easiness_test_data()
    model = train_nn(train_vector, train_score)
    accuracy = predict_nn(model, test_vector, test_score)
    print('nn accuracy:', accuracy)
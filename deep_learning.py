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


def train_nn(X, y):
  # Build NN model 
  model = Sequential()
  model.add(Dense(256, input_dim=3010, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(200, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(160, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(120, activation='relu'))
  model.add(Dropout(0.3))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()

  estimator = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=100)
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

  return "Avg train err: {}, Avg val err: {}".format(np.average(train_errs), np.average(val_errs))

if __name__ == "__main__":
    # Train NN on training data 
    X = text_pre_processing("train")
    y = bt_easiness_train_data()
    train_nn(X, y)
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
import json
from raw_data import *
from data_preprocessing import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from sklearn.model_selection import KFold
import data_preprocessing
import scipy as sp
import json
import imbalanced

"""
Train a neural network using keras.
"""
def neural_network(X_train, y_train, metrics=['mean_squared_error', 'mean_absolute_error'],
                   activation='relu', input_shape=(None, 2036), optimizer='adam', loss='mean_squared_error',
                   epochs=30, batch_size=64, verbose=1):
    model = Sequential()
    model.add(Dense(500, activation=activation, input_shape=input_shape))
    #model.add(Dropout(0.3))
    model.add(Dense(100, activation=activation))
    #model.add(Dropout(0.3))
    model.add(Dense(50, activation=activation))
    #model.add(Dropout(0.3))
    model.add(Dense(10, activation=activation))
    #model.add(Dropout(0.3))
    model.add(Dense(5, activation=activation))
    #model.add(Dropout(0.3))
    model.add(Dense(1))
    model.summary()
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    return model

def nn_train(type="train"):
  # Train NN on training data 
  all_data = create_new_features(type=type)
  train_vector = all_data[:, :-1]
  train_vector = train_vector.astype('float32')
  train_bt_easiness = all_data[:, -1]
  train_bt_easiness = train_bt_easiness.astype('float32')
  
  fold_num = 0
  kf = KFold(n_splits=5, shuffle=True)

  train_errs = []
  val_errs = []

  # Do k-fold validation here
  for train_index, test_index in kf.split(train_vector):
    print('Training on fold {}'.format(fold_num))
    X_train, X_test = train_vector[train_index], train_vector[test_index]
    y_train, y_test = train_bt_easiness[train_index], train_bt_easiness[test_index]

    #Resample the training data
    # X_train, y_train = imbalanced.resample(X_train, y_train, sample_type=sample_type)

    # Train the model on the train data
    model = neural_network(X_train, y_train, batch_size=64, input_shape=(X_train.shape[1],))

    results = nn_predict(model, X_train, y_train, type='train', k_val=fold_num)
    train_MSE = results['MSE']
  
    results = nn_predict(model, X_test, y_test, type='test', k_val=fold_num)
    val_MSE = results['MSE']

    fold_num += 1

    train_errs.append(float(train_MSE))
    val_errs.append(float(val_MSE))
  
  results = {
    'avg_train_MSE': np.average(train_errs),
    'avg_val_MSE': np.average(val_errs)
  }
  
  with open(f'keras_data/MSE_and_predictions_avg.json', 'w') as fp:
      json.dump(results, fp)

  return "Avg train err: {}, Avg val err: {}".format(np.average(train_errs), np.average(val_errs))

"""
Make predictions with a trained neural net model.

Input:
  model - NN model.
  test_vector - test vector input.
  k_val - the fold number if using k fold validation.
"""
def nn_predict(model, test_vector, test_bt_easiness, type="", k_val=0):
  predicted = model.predict(test_vector)
  MSE = mean_squared_error(predicted, test_bt_easiness)

  predicted = predicted.tolist()
  predicted = [float(x[0]) for x in predicted]

  results = {
    "MSE": float(MSE),
    "predicted_labels": predicted
  }

  # Save predictions and mean square error to json
  with open(f'keras_data/MSE_and_predictions_{k_val}_{type}.json', 'w') as fp:
      json.dump(results, fp)

  return results

if __name__ == "__main__":
  """
  # Testing data
  test_vector = create_new_features(type="test")
  test_vector = test_vector.astype('float32')
  test_bt_easiness = bt_easiness_test_data()
  test_bt_easiness = test_bt_easiness.astype('float32')
  """

  print(nn_train())
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
import json
from raw_data import *
from data_preprocessing import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN, LSTM, Embedding
from keras.utils import np_utils
from sklearn.model_selection import KFold
import data_preprocessing
import scipy as sp
import json
import imbalanced

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

"""
Train a neural network using keras.
"""
def neural_network(X_train, y_train, valid_set=[], metrics=['mean_squared_error', 'mean_absolute_error'],
                   activation='relu', input_shape=(None, 2036), optimizer='sgd', loss='mean_squared_error',
                   epochs=70
                   , batch_size=60, verbose=1):
    model = Sequential()
    model.add(Dense(500, activation=activation, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation=activation)) #kernel_regularizer=regularizers.L2(0.1)
    model.add(Dropout(0.3))
    model.add(Dense(100, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.summary()
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    history = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=valid_set, batch_size=batch_size, verbose=verbose)
    
    return history, model

def RNN(X_train, y_train, valid_set, metrics=['mean_squared_error', 'mean_absolute_error'],
                   activation='relu', input_shape=(None, 2036), optimizer='adam', loss='mean_squared_error',
                   epochs=70, batch_size=64, verbose=1):
  max_words = 2000
  max_len = 300
  tok = Tokenizer(num_words=max_words)
  tok.fit_on_texts(X_train)
  sequences = tok.texts_to_sequences(X_train)
  sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

  inputs = Input(name='inputs',shape=[max_len])
  layer = Embedding(max_words,50,input_length=max_len)(inputs)
  layer = LSTM(64)(layer)
  layer = Dense(256,name='FC1')(layer)
  layer = Activation('relu')(layer)
  layer = Dropout(0.5)(layer)
  layer = Dense(1,name='out_layer')(layer)
  model = Model(inputs=inputs,outputs=layer)
  model.compile(loss=loss,optimizer=optimizer,metrics=metrics)

  model.fit(sequences_matrix, y_train,batch_size=batch_size,epochs=epochs,
          validation_data=valid_set,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

  return model

def nn_train(type="train", kfold=True, test=False):
  # Train NN on training data 
  all_data = create_new_features(type=type, preprocessed=False)
  train_vector = all_data[:, :-1]
  train_vector = train_vector.astype('float32')
  train_bt_easiness = all_data[:, -1]
  train_bt_easiness = train_bt_easiness.astype('float32')
  
  fold_num = 0
  kf = KFold(n_splits=5, shuffle=True)

  train_errs = []
  val_errs = []

  if not kfold:
    X_train = train_vector
    y_train = train_bt_easiness

    all_data = create_new_features(type="test", preprocessed=False)
    test_vector = all_data[:, :-1]
    X_test = test_vector.astype('float32')
    test_bt_easiness = all_data[:, -1]
    y_test = test_bt_easiness.astype('float32')

    # Train the model on the train data
    history, model = neural_network(X_train, y_train, valid_set=(X_test, y_test), input_shape =(X_train.shape[1],))
    results_train = nn_predict(model, X_test, y_test, type='train', k_val=fold_num)

    '''if test == True:
      test_data = create_new_features(type="test", preprocessed=False)
      test_vector = test_data[:, :-1]
      test_bt_easiness = test_data[:, -1]
      results_test = nn_predict(model, test_vector, test_bt_easiness, type='test', k_val=fold_num)
      return results_test['MSE']'''

    train_MSE = results_train['MSE']
    
    return train_MSE

  # Do k-fold validation here
  for train_index, test_index in kf.split(train_vector):
    print('Training on fold {}'.format(fold_num))
    X_train, X_test = train_vector[train_index], train_vector[test_index]
    y_train, y_test = train_bt_easiness[train_index], train_bt_easiness[test_index]

    # Train the model on the train data
    history, model = neural_network(X_train, y_train, valid_set=(X_test, y_test), input_shape =(X_train.shape[1],))

    results_train = nn_predict(model, X_train, y_train, type='train', k_val=fold_num)
    train_MSE = results_train['MSE']
  
    results_val = nn_predict(model, X_test, y_test, type='test', k_val=fold_num)
    val_MSE = results_val['MSE']

    fold_num += 1

    train_errs.append(float(train_MSE))
    val_errs.append(float(val_MSE))
  
    # Plot the train vs validation error to check for overfitting
    print(history.history.keys())
    plt.plot(history.history['mean_squared_error'], color='red')
    plt.plot(history.history['val_mean_squared_error'], color = 'blue')
    plt.title('Model Accuracy for TF-IDF Vectorization')
    plt.ylabel('MSE Error')
    plt.xlabel('Number of Epochs')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('keras_err_curves2.png')
    plt.show()

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
  print(nn_train(type="train", kfold=True, test=False))
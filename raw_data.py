import pandas as pd 
import numpy as np

"""
Return all data available (test and train).
"""
def get_all_data():
  # read excel in data frame 
  df = pd.read_excel('CLEAR_Corpus_6.01.xlsx') 
  
  # convert a data frame to a Numpy 2D array 
  my_data = np.asarray(df) 
  return my_data

"""
Return train data.
"""
def train_data():
  all_data = get_all_data()
  mask = (all_data[:, -1] == "Train")
  return all_data[mask, :]

"""
Return test data.
"""
def test_data():
  all_data = get_all_data()
  mask = (all_data[:, -1] == "Test")
  return all_data[mask, :]

"""
Return the array of text excerpts for training.
"""
def text_train_data():
  array = train_data()
  return array[:, 14]

"""
Return the array of text excerpts for testing.
"""
def text_test_data():
  array = test_data()
  return array[:, 14]

"""
Return MPAA ratings for training (numbers).
"""
def mpaa_train_data():
  array = train_data()
  return array[:, 12]

"""
Return MPAA ratings for testing (numbers).
"""
def mpaa_test_data():
  array = test_data()
  return array[:, 12]

"""
Return BT-easiness score for training (numbers).
"""
def bt_easiness_all_data():
  array = get_all_data()
  return array[:, 22]


"""
Return the BT-easiness score for training (numbers).
"""
def bt_easiness_train_data():
  array = train_data()
  return array[:, 22]

"""
Return BT-easiness score for testing (numbers).
"""
def bt_easiness_test_data():
  array = test_data()
  return array[:, 22]

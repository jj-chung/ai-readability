import pandas as pd 
import numpy as np
import json

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
  # mask = (all_data[:, -1] == "Train")
  # return all_data[mask, :]

  np.random.seed(123)
  np.random.shuffle(all_data)
  split_idx = int(0.7 * all_data.shape[0])
  print(split_idx)

  return all_data[0:split_idx, :]

"""
Return test data.
"""
def test_data():
  all_data = get_all_data()
  # mask = (all_data[:, -1] == "Test")
  # return all_data[mask, :]

  np.random.seed(123)
  np.random.shuffle(all_data)
  split_idx = int(0.7 * all_data.shape[0])

  return all_data[split_idx: , :]


"""
Get ids for train and test data.
"""
def get_ids(type): 
  if type == "train":
    data = train_data()
    return data[:, 0]
  else:
    data = test_data()
    return data[:, 0]
  
"""
Return CEFR scores for training.
"""
def CEFR(type="train"):
    # Load data from json 
  # Opening JSON file
  f = open('cefrDATA.json', encoding="utf8")

  # returns JSON object as 
  # a dictionary
  data = json.load(f)

  # Iterating through the json
  # list
  X = []
  y = []

  ids = get_ids(type)

  for id in ids:
    dict = data[str(id)]
    X.append(dict["cefr_lvl"])
    y.append(dict["bt_easiness"])

  X = np.array(X).reshape(1, -1).T
  y = np.array(y).reshape(1, -1).T

  return X, y


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

if __name__ == "__main__":
  pass
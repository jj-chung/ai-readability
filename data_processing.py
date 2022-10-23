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
Only return train data.
"""
def train_data():
  all_data = get_all_data()
  mask = (all_data[:, -1] == "Train")
  return all_data[mask, :]

"""
Only return test data
"""
def test_data():
  all_data = get_all_data()
  mask = (all_data[:, -1] == "Test")
  return all_data[mask, :]

if __name__ == "__main__":
  print(test_data())
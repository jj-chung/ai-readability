import pandas as pd 
import numpy as np 

def get_all_data():
  # my_data = genfromtxt('CLEAR_Corpus_6.01.csv', delimiter=',', dtype=None)
  # read excel in data frame 
  df = pd.read_excel('CLEAR_Corpus_6.01.xlsx') 
  
  # convert a data frame to a Numpy 2D array 
  my_data = np.asarray(df) 
  return my_data

#heyyy
#helo


if __name__ == "__main__":
  print(get_all_data())
import pandas as pd 
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

"""
Only return the array of text excerpts for training
"""
def text_train_data():
  array = train_data()
  return array[:, 14]

"""
Only return the array of text excerpts for testing
"""
def text_test_data():
  array = test_data()
  return array[:, 14]

"""
Only return MPAA ratings for training (numbers)
"""
def mpaa_train_data():
  array = train_data()
  return array[:, 12]

"""
Only return MPAA ratings for testing (numbers)
"""
def mpaa_test_data():
  array = test_data()
  return array[:, 12]

"""
Convert training text to word vector
"""

def word_vectorizer():
  text = text_train_data()
  count_vect = sklearn.feature_extraction.text.CountVectorizer(max_features=2000)
  X_train_counts = count_vect.fit_transform(text)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  return X_train_tfidf

"""
Convert test text to word vector
"""

def word_vectorizer2():
  text = text_test_data()
  count_vect = sklearn.feature_extraction.text.CountVectorizer(max_features=2000)
  X_train_counts = count_vect.fit_transform(text)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  return X_train_tfidf


"""
Create 2 new features, average word length and average sentence length, using
the text excerpt data. 
"""
def create_new_features():
  # First, get all the data
  # Save the new data
  all_data = get_all_data()
  data_excerpts = all_data[:, 14]

  avg_word_length = []
  avg_sentence_length = []

  for excerpt in data_excerpts:
    # Compute average word length for the excerpt
    words = excerpt.split()
    print(words)

    total_avg = sum( map(len, words) ) / len(words)
    avg_word_length.append(total_avg)

    # Compute average sentence length for the excerpt
    sentences = excerpt.split(".")
    total_avg = sum( map(len, sentences) ) / len(sentences)
    avg_sentence_length.append(total_avg)

  

  

if __name__ == "__main__":
  # print(test_data())
  # print(train_data())
  # print(text_train_data())
  create_new_features()

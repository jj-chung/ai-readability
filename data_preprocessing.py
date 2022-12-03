from raw_data import *
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
import contractions
import spacy
from spacy_syllables import SpacySyllables
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random

"""
Convert text to word vector.

Input: Flag for determining whether data is test or train.
"""
def word_vectorizer_train(type="train"):
  text = None

  text = text_pre_processing(type)

  count_vect = sklearn.feature_extraction.text.CountVectorizer(max_features=2000)
  X_train_counts = count_vect.fit_transform(text)
  #print('vect_length: ', X_train_counts.shape[0], X_train_counts.shape[1], X_train_counts)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  return X_train_tfidf


def word_vectorizer_keras(type="train"):
  text = None

  text = text_pre_processing(type)

  vec = TfidfVectorizer(max_features=2000)
  tfidf_mat = vec.fit_transform(text).toarray()

  return tfidf_mat

"""
Further pre-process the data to exclude stop words. Also use text tokenization
to reduce text to tokens.

Input: Flag for determining whether data is test or train.
Output: Preprocessed text data.
"""
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
      return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def text_pre_processing(type="train"):
  data = None

  if type == "train":
    data = text_train_data()
  elif type == "test":
    data = text_test_data()

  for i in range(data.shape[0]):
    if i % 100 == 0:
      print(i)

    excerpt = data[i]

    # Remove contractions from data
    excerpt = contractions.fix(excerpt)

    # Convert all text to be lower case 
    excerpt = excerpt.lower()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = excerpt.split(' ')
    excerpt = " ".join([word for word in words if word not in stop_words])

    data[i] = excerpt

  return data

def mpaa_pre_processing(type="train"):
  data = None

  if type == "train":
    data = mpaa_train_data()
  elif type == "test":
    data = mpaa_test_data()
  
  # merge R and PG-13
  return np.array(list(map(lambda x: 3 if x == 4 else x, data)))

"""
Create 2 new features, average word length and average sentence length, using
the text excerpt data. 
"""
def create_new_features(type="train", baseline=True):
  # First, get all the data
  # Save the new data
  data_excerpts = text_pre_processing(type)

  avg_word_length = []
  avg_sentence_length = []
  unique_word_ct = []
  avg_syllables = []

  nlp = spacy.load("en_core_web_sm")
  nlp.add_pipe("syllables", after="tagger")
  nlp.add_pipe('sentencizer')

  for i in range(data_excerpts.shape[0]):
    excerpt = data_excerpts[i]

    if i % 100 == 0:
      print(i)

    # Compute average word length for the excerpt
    doc = nlp(excerpt)
    words = [token.text for token in doc if (not token.is_punct and token.text != '\n')]
    syllables_list = [token._.syllables_count for token in doc if (not token.is_punct and token.text != '\n')]

    total_avg = sum( map(len, words) ) / len(words)
    avg_word_length.append(total_avg)

    # Compute average sentence length for the excerpt
    sentences = [sent for sent in doc.sents]

    total_avg = sum( map(len, sentences) ) / len(sentences)
    avg_sentence_length.append(total_avg)

    # Consider the number of uncommon words in the text
    # uncommon_words_ct = 

    # Consider the number of unique words in the text
    unique_word_ct.append(len(set(words)))

    # Consider the average number of syllables
    syllables_list = [s_count for s_count in syllables_list if s_count is not None]
    avg_syllables.append(np.average(np.array(syllables_list)))

  # Create a numpy array with the average word length as a column,
  # the average sentence length as a column,
  # and bt-easiness as the third column
  bt_easiness = None
  
  if type == "train":
    bt_easiness = bt_easiness_train_data()
  elif type == "test":
    bt_easiness = bt_easiness_test_data()

  # For our baseline model, we only consider average word length as a feature 
  # and average sentence length as a feature.
  if baseline:
    features_arr = np.column_stack((avg_word_length, avg_sentence_length, bt_easiness))
    print(features_arr.shape)
    return features_arr
  
  features_arr = np.column_stack((avg_word_length, avg_sentence_length,  
    unique_word_ct, avg_syllables, bt_easiness))
  print(features_arr.shape)
  return features_arr

def find_stopwords_overlap():
  set1 = set(stopwords.words("english"))
  en = spacy.load("en_core_web_sm")
  set2 = set(en.Defaults.stop_words)

  return len(set1.intersection(set2))

def color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    nltk_stopwords = stopwords.words("english")
    if word in nltk_stopwords:
      return "hsl(110, 100%, 0%)"
    else:
      return "hsl(110, 100%%, %d%%)" % 40

def visualize_stopwords():
  text_arr = text_train_data()
  combined_text = ""

  for elem in text_arr:
    combined_text += elem

  wordcloud = WordCloud(width = 800, height = 600,
                background_color ='white',
                stopwords = [],
                min_font_size = 10).generate(combined_text)

  wordcloud = wordcloud.recolor(color_func=color_func, random_state=3)

  # Plot the WordCloud image                      
  plt.figure(figsize = (8, 8), facecolor = None)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout(pad = 0)
  
  plt.show()

if __name__ == "__main__":
  visualize_stopwords()
  
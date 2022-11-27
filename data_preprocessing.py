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

"""
Convert text to word vector.

Input: Flag for determining whether data is test or train.
"""
def word_vectorizer_train(type="train"):
  text = None

  if type == "train":
    text = text_train_data()
  elif type == "test":
    text = text_test_data()

  count_vect = sklearn.feature_extraction.text.CountVectorizer(max_features=2000)
  X_train_counts = count_vect.fit_transform(text)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  return X_train_tfidf

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
    excerpt = data[i]

    # Remove contractions from data
    excerpt = contractions.fix(excerpt)

    # Convert all text to be lower case 
    excerpt = excerpt.lower()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = excerpt.split()
    excerpt = "".join([word for word in words if word not in stop_words])

    data[i] = excerpt

  # Create feature vector based on word counts
  # Let vocabulary consist of stems and lemmas
  tfidf = TfidfVectorizer(tokenizer=LemmaTokenizer(), 
                         max_features=2000)

  data = tfidf.fit_transform(data)

  return data

"""
Create 2 new features, average word length and average sentence length, using
the text excerpt data. 
"""
def create_new_features(type="train"):
  # First, get all the data
  # Save the new data
  data_excerpts = text_pre_processing(type)

  avg_word_length = []
  avg_sentence_length = []

  for excerpt in data_excerpts:
    # Compute average word length for the excerpt
    words = excerpt.split()

    total_avg = sum( map(len, words) ) / len(words)
    avg_word_length.append(total_avg)

    # Compute average sentence length for the excerpt
    sentences = excerpt.split(".")
    total_avg = sum( map(len, sentences) ) / len(sentences)
    avg_sentence_length.append(total_avg)

  # Create a numpy array with the average word length as a column,
  # the average sentence length as a column,
  # and bt-easiness as the third column
  bt_easiness = None
  if type == "train":
    bt_easiness = bt_easiness_train_data()
  elif type == "test":
    bt_easiness = bt_easiness_test_data()

  features_arr = np.column_stack((avg_word_length, avg_sentence_length, bt_easiness))
  return features_arr

# if __name__ == "__main__":
  # print(test_data())
  # print(train_data())
  # print(text_train_data())
  # print(create_new_features(train_data))
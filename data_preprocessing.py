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
import scipy as sp
from sklearn import preprocessing
import json
from json import JSONEncoder
import h5py
from pathlib import Path
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

# from datasets import Dataset
# from transformers import AutoTokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Applies the tokenizer for bert to an example text-- includes [CLS] token (I think?)
def tokenize(data):
    return tokenizer(data["text"], padding="max_length", truncation=True)

# Gets data preprocessed into an array of label, token dicts for Bert
def bert_pre_processing(type="train"):
  data = None
  if type == "train":
    data = text_train_data()
  elif type == "test":
    data = text_test_data()
  
  # merge labels and data 
  labels = mpaa_pre_processing(type=type)

  def gen(data, labels):
    for i, label in enumerate(labels):
      yield {"text": data[i], "label": label}
  data_set = Dataset.from_generator(gen, gen_kwargs={"data": data, "labels": labels})

  # structure data like: https://huggingface.co/docs/transformers/training#train-with-pytorch-trainer
  # data_set = [{'label': label, 'text': tokenize()} for i, label in enumerate(labels)]
  return data_set.map(tokenize, batched=True)

"""
Convert text to word vector.

Input: Flag for determining whether data is test or train.
"""
def word_vectorizer(type="train"):
  """
  count_vect = sklearn.feature_extraction.text.CountVectorizer(max_features=2000)
  X_train_counts = count_vect.fit_transform(text)
  #print('vect_length: ', X_train_counts.shape[0], X_train_counts.shape[1], X_train_counts)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  return X_train_tfidf
  """
  if type=="train":
    data_excerpts = text_train_data()
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(data_excerpts)
  else:
    data_excerpts = text_test_data()
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(data_excerpts)

  return X

"""
Use word2vec embedding/representation to convert data into vectors instead of
tfidf.
"""
def word_vectorizer_word2vec(type="train"):
    text_data = text_pre_processing()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('sentencizer')

    sentences = []

    for i, excerpt in enumerate(text_data):
      doc = nlp(excerpt)
      td = TaggedDocument([token.text for token in doc if (not token.is_punct and token.text != '\n')], [i])
      sentences.append(td)

    train_tagged_list = sentences
          
    model_dbow = Doc2Vec(dm=0, vector_size=1000, negative=5, hs=0, min_count=2, sample = 0)
    model_dbow.build_vocab(train_tagged_list)
    
    for epoch in range(30):
        random.shuffle(train_tagged_list)
        model_dbow.train(train_tagged_list, total_examples=len(train_tagged_list), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    X_train = vec_for_learning(model_dbow, sentences)

    return X_train
  
def vec_for_learning(model, sentences):
    sents = sentences
    regressors = [model.infer_vector(doc.words) for doc in sents]
    return regressors

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
  if type == "train":
    data = text_train_data()
  elif type == "test":
    data = text_test_data()

  for i in range(data.shape[0]):
    if i % 200 == 0:
      print(f'Excerpt cleaned: {i}')

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
def create_new_features(type="train", baseline=False, preprocessed=False):
  # Load data if specified and the file exists
  my_file = Path("features_arr.h5")
  if my_file.is_file() and preprocessed:
    f = h5py.File('features_arr.h5','r')
    finalNumpyArray = f.get('features_arr') #.value
    return finalNumpyArray
  
  # First, get all the data
  # Save the new data
  data_excerpts = text_pre_processing(type)

  avg_word_length = []
  avg_sentence_length = []
  non_unique_word_ct = []
  unique_word_ct = []
  avg_syllables = []
  num_punct_arr = []
  orig_len = []
  lemma_ct = []
  lemma_len = []
  len_ratio = []
  lemma_ratio = []
  max_word_len = []
  words_per_sentence = []

  # punctuation
  commas = []
  semicolons = []
  exclamations = []
  questions = []
  quotes = []
  periods = []

  pos_dict = {}
  for key in ['ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ',	'X',
    'NOUN',	'DET', 'PROPN', 'NUM', 'VERB', 'PART', 'PRON', 'SCONJ', 'SPACE']:
    pos_dict[key] = []

  tense_dict = {}
  for key in ['Fut','Imp', 'Past', 'Pqp', 'Pres']:
    tense_dict[key] = []

  verbForm_dict = {}
  for key in ['Conv', 'Fin', 'Gdv', 'Ger', 'Inf', 'Part', 'Sup', 'Vnoun']:
    verbForm_dict[key] = []

  overall_word_dicts = [pos_dict, tense_dict, verbForm_dict]

  nlp = spacy.load("en_core_web_sm")
  nlp.add_pipe("syllables", after="tagger")
  nlp.add_pipe('sentencizer')

  for i in range(data_excerpts.shape[0]):
    excerpt = data_excerpts[i]

    if i % 200 == 0:
      print(f'Excerpts Preprocessed: {i}')

    doc = nlp(excerpt)

    # Add the length of the excerpt before preprocessing
    ex_length = len(excerpt)
    orig_len.append(ex_length)

    # Add number of lemmas and length of text if we only count lemmas
    lemmas = [token.lemma_ for token in doc if (not token.is_punct and token.text != '\n')]
    lemma_ct.append(len(lemmas))
    lemma_length = len(" ".join(lemmas))
    lemma_len.append(lemma_length)
    len_ratio.append(lemma_length / ex_length)
    lemma_ratio.append(len(lemmas) / ex_length)

    # Compute average word length for the excerpt
    words = [token.text for token in doc if (not token.is_punct and token.text != '\n')]
    syllables_list = [token._.syllables_count for token in doc if (not token.is_punct and token.text != '\n')]
    punctuation_list = [token.text for token in doc if (token.is_punct and token.text != '\n')]

    commas.append(excerpt.count(","))
    semicolons.append(excerpt.count(";"))
    exclamations.append(excerpt.count("!"))
    questions.append(excerpt.count("?"))
    quotes.append(excerpt.count('"'))
    periods.append(excerpt.count('.'))

    total_avg = sum( map(len, words) ) / len(words)
    avg_word_length.append(total_avg)

    # Maximum word length
    max_word_len.append(len(max(words, key =len)))

    # Compute average sentence length for the excerpt
    sentences = [sent for sent in doc.sents]
    total_avg = sum( map(len, sentences) ) / len(sentences)
    avg_sentence_length.append(total_avg)
    words_per_sentence.append(len(words)/len(sentences))

    pos_dict_excerpt = {}
    verbForms_excerpt = {}
    tenses_excerpt = {}

    word_dicts = [pos_dict_excerpt, verbForms_excerpt, tenses_excerpt]

    # Consider the part of speech vector
    word_pos = [token.pos_ for token in doc]

    # Consider the morphology of the word (how the stem (lemma) is converted to 
    # its existing form in the text)
    word_verbForm = [token.morph.get('VerbForm') for token in doc]
    word_verbForm = ['' if form == [] else form[0] for form in word_verbForm]
    word_tense = [token.morph.get('Tense') for token in doc]
    word_tense = ['' if tense == [] else tense[0] for tense in word_tense]

    # List of all word attributes
    word_lists = [word_pos, word_verbForm, word_tense]
    
    # Create a dictionary from tense/verbform/pos to counts
    for j, word_dict in enumerate(word_dicts):
      word_list = word_lists[j]
      for attribute in word_list:
        if attribute not in word_dict:
          word_dict[attribute] = 1
        else:
          word_dict[attribute] += 1

    # Append each counts dictionary respective feature vector
    for k, overall_word_dict in enumerate(overall_word_dicts):
      word_dict = word_dicts[k]

      for key in overall_word_dict:
        if key in word_dict:
          count = word_dict[key] / len(doc)
          overall_word_dict[key].append(count)
        else:
          overall_word_dict[key].append(0)

    # The amount of puncutation in the text
    num_punct = len(punctuation_list)
    num_punct_arr.append(num_punct)

    # Consider the number of unique words in the text
    unique_word_ct.append(len(set(words)))
    non_unique_word_ct.append(len(words))

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
    return features_arr
  
  features_arr = np.column_stack((avg_word_length, 
                                  avg_sentence_length, 
                                  num_punct_arr,
                                  unique_word_ct, 
                                  non_unique_word_ct,
                                  avg_syllables, 
                                  orig_len,
                                  lemma_ct,
                                  lemma_len,
                                  lemma_ratio,
                                  len_ratio,
                                  max_word_len))


  # Append additional spacy NLP characteristics
  for word_dict in overall_word_dicts:
    for attribute in word_dict:
      features_arr = np.column_stack((features_arr, word_dict[attribute]))


  # Normalize X-features to have zero mean and unit variance
  scaler = preprocessing.StandardScaler().fit(features_arr)
  features_arr = scaler.transform(features_arr)

  # Append all the word2vec vectorizations
  word_vectors = word_vectorizer_word2vec(type=type)
  features_arr = np.column_stack((word_vectors, features_arr))

  # Append all the TFIDF vectorizations
  #word_vectors = word_vectorizer(type=type).toarray()
  #features_arr = np.column_stack((word_vectors, features_arr))

  # Tack on y-data, i.e. bt-easiness
  features_arr = np.column_stack((features_arr, bt_easiness))

  # Save the features array in a json file
  '''f = h5py.File('features_arr.h5', 'w')
  f.create_dataset("features_arr", data=features_arr.astype('float32'))'''

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
  
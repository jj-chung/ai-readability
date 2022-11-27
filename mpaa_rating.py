import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import raw_data
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

def trained_NB_model(word_vectors, mpaa_ratings, test_train, test_target):
  model = MultinomialNB().fit(word_vectors, mpaa_ratings)
  predict_test = model.predict(test_train)
  return predict_test, np.mean(predict_test == test_target)*100

def trained_KNN_model(word_vectors, mpaa_ratings, test_train, test_target):
  # List Hyperparameters that we want to tune
  leaf_size = list(range(1,9))
  n_neighbors = list(range(1,9))
  p = [1, 2]

  # Convert to dictionary
  hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

  # Create new KNN object
  knn = KNeighborsClassifier(n_neighbors=8)

  # Use GridSearch
  clf = GridSearchCV(knn, hyperparameters, cv=2)

  # Fit the model
  best_model = clf.fit(word_vectors, mpaa_ratings)

  # Print The value of best Hyperparameters
  print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
  print('Best p:', best_model.best_estimator_.get_params()['p'])
  print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

  predict_test = clf.predict(test_train)
  # return the best model
  #return accuracy rate
  return predict_test, np.mean(predict_test == test_target)*100


def plot_scores(clf):
  scores = clf.cv_results_

if __name__ == "__main__":
  '''word_vectors = [["hello"], ["hi"], ["hey"], ["amaa"], ["damn"], ["six"]]
  categories = ['rec.motorcycles', 'sci.electronics',
                'comp.graphics', 'sci.med']

  train_data = fetch_20newsgroups(subset='train',
                                  categories=categories, shuffle=True, random_state=42)
  tfidf_transformer = TfidfTransformer()
  count_vect = CountVectorizer()
  X_train_counts = count_vect.fit_transform(train_data.data)
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  #clf, best_model = trained_KNN_model(X_train_tfidf, train_data.target)
  print(train_data.target)
  mpaa_ratings = data_processing.mpaa_train_data()
  print(mpaa_ratings.astype(int))'''
  #https://iq.opengenus.org/text-classification-using-k-nearest-neighbors/

  words = raw_data.word_vectorizer()
  mpaa_ratings = raw_data.mpaa_train_data()
  test_train = raw_data.word_vectorizer2()
  test_target = raw_data.mpaa_test_data()
  predicted, accuracy = trained_KNN_model(words, mpaa_ratings.astype(int), test_train, test_target.astype(int))
  print(predicted)
  y_true = raw_data.mpaa_test_data().astype(int)
  print(y_true)
  '''matrix = sklearn.metrics.confusion_matrix(y_true, predicted)
  label_font = {'size': '18'}
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["G", "PG", "PG-13", "R"])
  cm_display.plot()
  plt.rcParams.update({'font.size': 33})
  plt.show()'''
  print("ur knn accuracy is:", accuracy)
  predicted_NB, accuracy_NB = trained_NB_model(words, mpaa_ratings.astype(int), test_train, test_target.astype(int))
  print("ur NB accuracy is: ", accuracy_NB)
  #2nd confusion matrix for NB
  matrix = sklearn.metrics.confusion_matrix(y_true, predicted_NB)
  label_font = {'size': '18'}
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["G", "PG", "PG-13", "R"])
  cm_display.plot()
  plt.rcParams.update({'font.size': 33})
  plt.show()
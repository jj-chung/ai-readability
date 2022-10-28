import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
import data_processing
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def trained_KNN_model(word_vectors, mpaa_ratings, test_train, test_target):
  # List Hyperparameters that we want to tune
  leaf_size = list(range(1,3))
  n_neighbors = list(range(1,3))
  p = [1, 2]

  # Convert to dictionary
  hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

  # Create new KNN object
  knn = KNeighborsClassifier()

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
  return np.mean(predict_test == test_target)*100


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

  words = data_processing.word_vectorizer()
  mpaa_ratings = data_processing.mpaa_train_data()
  test_train = data_processing.word_vectorizer2()
  test_target = data_processing.mpaa_test_data()
  #[1, 0, 1, 1, 0, 1]
  print(test_train, "\n \n")
  print(words)
  accuracy = trained_KNN_model(words, mpaa_ratings.astype(int), test_train, test_target.astype(int))
  print(accuracy)
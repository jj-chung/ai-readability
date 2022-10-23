import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def trained_KNN_model(word_vectors, mpaa_ratings):
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

  # return the best model 
  return clf, best_model

def plot_scores(clf):
  scores = clf.cv_results_

if __name__ == "__main__":
  word_vectors = [["hello"], ["hi"], ["hey"]]
  mpaa_ratings = [1, 0, 1]
  clf, best_model = trained_KNN_model(word_vectors, mpaa_ratings)
  print(clf.cv_results_)
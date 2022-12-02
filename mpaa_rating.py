import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing
import imbalanced
import json

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

"""
Training the SVM model using gridsearch. 

Input:
word_vectors - tfidf vectorization for training data
mpaa_ratings - target variable/training labels (Numbers from 1 to 4, G to R)
grid_search - Boolean flag for whether or not to perform grid search here
"""
def train_SVM_model(word_vectors, mpaa_ratings, grid_search=True):
  if grid_search:
    # Use gridsearch to find the optimal set of parameters
    param_grid = ({'C':[0.1,1,100],
                  'kernel':['rbf','poly','sigmoid','linear'],
                  'degree':[1,2,3]})
  else:
    param_grid = ({'C':[1],
                  'kernel':['rbf'],
                  'degree':[2]})

  # Create SVM and GridSearch object
  svm = sklearn.svm.SVC()
  grid = GridSearchCV(svm, param_grid)

  # Search over the grid for optimal parameters for SVM
  model = grid.fit(word_vectors, mpaa_ratings)
  
  # Save optimal parameters to JSON file
  optim_params = grid.best_estimator_.get_params()

  with open('optimal_SVM_params.json', 'w') as fp:
    json.dump(optim_params, fp)

  return model

"""
Train the Naive Bayes model.
Input:
  word_vectors - tfidf vectorization for training data
  mpaa_ratings - target variable/training labels (Numbers from 1 to 4, G to R)
"""
def train_NB_model(word_vectors, mpaa_ratings):
  model = MultinomialNB().fit(word_vectors, mpaa_ratings)

  return model

"""
Train an Adaboost model.
Input:
  word_vectors - tfidf vectorization for training data
  mpaa_ratings - target variable/training labels (Numbers from 1 to 4, G to R)
"""
def train_Adaboost_model(word_vectors, mpaa_ratings, estimator = None):
    model = AdaBoostClassifier(base_estimator = estimator, n_estimators=100, random_state=42).fit(word_vectors, mpaa_ratings)
    
    return model

"""
Train the KNN model.
Input:
  word_vectors - tfidf vectorization for training data
  mpaa_ratings - target variable/training labels (Numbers from 1 to 4, G to R)
"""
def train_KNN_model(word_vectors, mpaa_ratings):
  # List hyperparameters that we want to tune
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
  model = clf.fit(word_vectors, mpaa_ratings)

  # Save optimal parameters to JSON file
  optim_params = model.best_estimator_.get_params()

  with open('optimal_SVM_params.json', 'w') as fp:
    json.dump(optim_params, fp)

  return model

"""
Predict for SVM/NB/KNN/Adaboost model.

Input:
test_vectors - input tfidf vectors for the test data
test_target - MPAA rating labels for the test data (Numbers from 1 to 4, G to R)
model_type - string specifying whether the model is SVM, Naive Bayes, KNN, Adaboost, or something else
   MUST be included for optimal parameters to be saved correctly/not overwritten.

IMPORTANT: DO NOT run this function until all training and validation has been completed.
For final prediction only.
"""
def predict_model(model, test_vectors, test_target, model_type="SVM"):
  # Predicted MPAA ratings
  predicted_ratings = model.predict(test_vectors)

  # Compute the number of predictions which were correct for PG-13 movies
  PG_13_indices = predicted_ratings == 3
  num_correct_PG_13 = sum(predicted_ratings[PG_13_indices] == test_target[PG_13_indices]) 
  total_PG_13 = test_target[PG_13_indices].shape([0])
  PG_13_correct = num_correct_PG_13 / total_PG_13

  eval_metrics = {
      "predicted": predicted_ratings, 
      "accuracy_score" : accuracy_score(test_target, predicted_ratings), 
      "f1_score": f1_score(test_target, predicted_ratings, average='weighted'),
      "PG_13_correct": PG_13_correct
  }
      
  with open('test_eval_metrics_{}.json'.format(model_type), 'w') as fp:
    json.dump(eval_metrics, fp)

  return eval_metrics

"""
Given data and a model type, it will train and validate the model.
Displays confusion matrices for each of the k validation sets, and the 
"average" confusion matrix (i.e. the average value for each of the cells)

Input:
  train_vector - tfidf vectors for train data
  train_labels - MPAA ratings for train data
  conf_matrix - Boolean flag for whether or not to display the confusion matrices

Output:
  None.
"""
def train_model(train_vector, train_labels, model_type, conf_matrix=True):
  if model_type == 'knn':
    model = train_KNN_model(train_vector, train_labels.astype(int))
  elif model == 'nb':
    model = train_NB_model(train_vector, train_labels.astype(int))
  elif model_type == 'svm':
    # Make sure to update grid search flag as needed
    model = train_SVM_model(train_vector, train_labels.astype(int), grid_search=False)
  elif model_type == 'adaboost':
    model = train_Adaboost_model(train_vector, train_labels.astype(int))
  elif model == 'adaboost_nb': 
    model = train_Adaboost_model(train_vector, train_labels.astype(int), estimator = MultinomialNB())

  if conf_matrix:
    matrix = sklearn.metrics.confusion_matrix(test_labels, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["G", "PG", "Mature"])
    cm_display.plot()
    plt.rcParams.update({'font.size': 33})
    plt.title(conf_matrix)
    plt.show()

"""
Resample the data to treat imbalanced classes for MPAA Ratings.
Input:
train_vector - tfidf word vectors for training examples
train_labels - MPAA ratings for training examples
model_type - 
"""
def resampling_method(train_vector, train_labels, model_type="svm"):
  #---Resampled Data---
  x_rus, y_rus = imbalanced.RUS(train_vector, train_labels)
  x_tl, y_tl = imbalanced.TLinks(train_vector, train_labels)
  x_ros, y_ros = imbalanced.ROS(train_vector, train_labels)
  x_sm, y_sm = imbalanced.SMOTE_Reg(train_vector, train_labels)
  x_smt, y_smt = imbalanced.SMOTE_TL(train_vector, train_labels)

  all_data = [(train_vector, train_labels), (x_rus, y_rus), (x_tl, y_tl), (x_ros, y_ros), (x_sm, y_sm), (x_smt, y_smt)]

  x_en, y_en = imbalanced.ENN(train_vector, train_labels)

  DATA_SETS = ['Imbalanced', 'RUS', 'TomekLinks', 'ROS', 'SMOTE', 'SMOTETomek']
  for i, data_set in enumerate(DATA_SETS):
    print(f"ur {data_set} ab/nb accuracy is:", train_model(all_data[i][0], all_data[i][1], test_vector, test_labels, model_type, conf_matrix = data_set))

if __name__ == "__main__":
  #---Vectorizing Data---
  train_vector = data_preprocessing.word_vectorizer_train(type="train")
  train_labels = data_preprocessing.mpaa_pre_processing(type="train")
  test_vector = data_preprocessing.word_vectorizer_train(type="test")
  test_labels = data_preprocessing.mpaa_pre_processing(type="test")

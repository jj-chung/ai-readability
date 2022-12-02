import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing
import imbalanced
import json

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
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
def train_SVM_model(word_vectors, mpaa_ratings, data_type, grid_search=True, do_k_fold=True):
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
  clf = GridSearchCV(svm, param_grid)

  # Fit the model. If k-fold validation is specified, call the k-fold validation
  # function fit and predict k times.
  if do_k_fold:
    k_fold_validation(clf, word_vectors, mpaa_ratings, "SVM", data_type)
  else:
    model = clf.fit(word_vectors, mpaa_ratings)

    # Save optimal parameters to JSON file
    optim_params = model.best_estimator_.get_params()

    with open('mpaa_data/optimal_SVM_params.json', 'w') as fp:
      json.dump(optim_params, fp)

    return model

"""
Train the Naive Bayes model.
Input:
  word_vectors - tfidf vectorization for training data
  mpaa_ratings - target variable/training labels (Numbers from 1 to 4, G to R)
"""
def train_NB_model(word_vectors, mpaa_ratings, data_type, do_k_fold=True):
  if do_k_fold:
    k_fold_validation(MultinomialNB(), word_vectors, mpaa_ratings, "NB", data_type)
  else:
    model = MultinomialNB().fit(word_vectors, mpaa_ratings)
    return model

"""
Train an Adaboost model.
Input:
  word_vectors - tfidf vectorization for training data
  mpaa_ratings - target variable/training labels (Numbers from 1 to 4, G to R)
"""
def train_Adaboost_model(word_vectors, mpaa_ratings, data_type, estimator = None, do_k_fold=True):
  ABClassifier = AdaBoostClassifier(base_estimator = estimator, n_estimators=100, random_state=42)
  
  if do_k_fold:
    k_fold_validation(model, word_vectors, mpaa_ratings, "AdaBoost", data_type)
  else:
    model = ABClassifier.fit(word_vectors, mpaa_ratings)
    return model
    

"""
Train the KNN model.
Input:
  word_vectors - tfidf vectorization for training data
  mpaa_ratings - target variable/training labels (Numbers from 1 to 4, G to R)
"""
def train_KNN_model(word_vectors, mpaa_ratings, data_type, do_k_fold=True):
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

  # Fit the model. If k-fold validation is specified, call the k-fold validation
  # function fit and predict k times.
  if do_k_fold:
    k_fold_validation(clf, word_vectors, mpaa_ratings, "KNN", data_type)
  else:
    model = clf.fit(word_vectors, mpaa_ratings)

    # Save optimal parameters to JSON file
    optim_params = model.best_estimator_.get_params()

    with open('mpaa_data/optimal_KNN_params.json', 'w') as fp:
      json.dump(optim_params, fp)

  return model

"""
Perform k-fold validation for a model, a training dataset, and a k value.
Input:
  model - the model being trained on each run of k-fold-validation
  train_vectors - the training dataset tfidf vectors
  train_ratings - the training dataset MPAA ratings
"""
def k_fold_validation(clf, train_vectors, train_ratings, model_type, data_type, k=5):
  X = train_vectors
  y = train_ratings

  kf = KFold(n_splits=k)
  fold_num = 0
  avg_eval_metrics = {
      "accuracy_score" : [],
      "f1_score": [],
      "Mature_correct": []
  }

  for train_index, test_index in kf.split(X):
    print(f'Training on fold {fold_num}')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model on the train data
    model = clf.fit(X_train, y_train)

    if model_type in ["SVM", "KNN"]:
      # Save optimal parameters to JSON file
      optim_params = model.best_estimator_.get_params()
      with open(f'mpaa_data/optimal_{model_type}_params_{k}_{data_type}.json', 'w') as fp:
        json.dump(optim_params, fp)

    # Evaluate the model on the test data
    eval_metrics = predict_model(model, X_test, y_test, data_type, model_type=model_type, conf_matrix=True, k_val=fold_num)

    for key in eval_metrics:
      if key in avg_eval_metrics:
        avg_eval_metrics[key].append(eval_metrics[key])

    # Add to fold variable 
    fold_num += 1

  # Save avg evaluation metrics to json
  for key in avg_eval_metrics:
    values = avg_eval_metrics[key]
    avg_eval_metrics[key] = np.average(values)

  # Save optimal parameters to JSON file
  with open(f'mpaa_data/avg_eval_metrics_{model_type}_{data_type}.json', 'w') as fp:
    json.dump(avg_eval_metrics, fp)

  return avg_eval_metrics

"""
Predict for SVM/NB/KNN/Adaboost model.

Input:
test_vectors - input tfidf vectors for the test data
test_target - MPAA rating labels for the test data (Numbers from 1 to 4, G to R)
model_type - string specifying whether the model is SVM, Naive Bayes, KNN, Adaboost, or something else
   MUST be included for optimal parameters to be saved correctly/not overwritten.
conf_matrix - whether or not a confusion matrix should be generated
k - Value of k if prediction is being used for k-fold validation

IMPORTANT: DO NOT run this function on actual test data until all training and 
validation has been completed.
"""
def predict_model(model, test_vectors, test_target, data_type, model_type="SVM", conf_matrix=True, k_val="N/A"):
  # Predicted MPAA ratings
  predicted_ratings = model.predict(test_vectors)

  # Compute the number of predictions which were correct for Mature movies
  Mature_indices = predicted_ratings == 3
  num_correct_Mature = sum(predicted_ratings[Mature_indices] == test_target[Mature_indices]) 
  total_Mature = len(test_target == 3)
  Mature_correct = num_correct_Mature / total_Mature

  # Compute the evaluation metrics
  eval_metrics = {
      "predict_labels" : predicted_ratings.tolist(),
      "accuracy_score" : float(accuracy_score(test_target, predicted_ratings)), 
      "f1_score": float(f1_score(test_target, predicted_ratings, average='weighted')),
      "Mature_correct": float(Mature_correct)
  }
      
  with open(f'mpaa_data/test_eval_metrics_{model_type}_{k_val}.json', 'w') as fp:
    json.dump(eval_metrics, fp)

  # Create a confusion matrix
  if conf_matrix:
    matrix = confusion_matrix(test_target, predicted_ratings)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["G", "PG", "Mature"])
    cm_display.plot()
    plt.rcParams.update({'font.size': 20})
    plt.title(f"Confusion Matrix for {model_type}")
    plt.savefig(f"images/Confusion_Matrix_{model_type}_k={k_val}_{data_type}")

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
def train_model(train_vector, train_labels, model_type, data_type="balanced"):
  if model_type == 'knn':
    model = train_KNN_model(train_vector, train_labels.astype(int), data_type)
  elif model_type == 'nb':
    model = train_NB_model(train_vector, train_labels.astype(int), data_type)
  elif model_type == 'svm':
    # Make sure to update grid search flag as needed
    model = train_SVM_model(train_vector, train_labels.astype(int), data_type, grid_search=False)
  elif model_type == 'adaboost':
    model = train_Adaboost_model(train_vector, train_labels.astype(int))
  elif model_type == 'adaboost_nb': 
    model = train_Adaboost_model(train_vector, train_labels.astype(int), data_type, estimator = MultinomialNB())

  return model

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

  x_en, y_en = imbalanced.ENN(train_vector, train_labels)

  all_data = [(train_vector, train_labels), (x_rus, y_rus), (x_tl, y_tl), (x_en, y_en),
    (x_ros, y_ros), (x_sm, y_sm), (x_smt, y_smt)]

  DATA_SETS = ['Imbalanced', 'RUS', 'TomekLinks', 'ENN', 'ROS', 'SMOTE', 'SMOTETomek']
  for i, data_set in enumerate(DATA_SETS):
    train_model(all_data[i][0], all_data[i][1], model_type, data_type=data_set)

if __name__ == "__main__":
  #---Vectorizing Data---
  train_vector = data_preprocessing.word_vectorizer_train(type="train")
  train_labels = data_preprocessing.mpaa_pre_processing(type="train")
  test_vector = data_preprocessing.word_vectorizer_train(type="test")
  test_labels = data_preprocessing.mpaa_pre_processing(type="test")

  resampling_method(train_vector, train_labels, "nb")

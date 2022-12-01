import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import data_preprocessing
import imbalanced
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

DATA_SETS = ['Imbalanced', 'RUS', 'TomekLinks', 'ROS', 'SMOTE', 'SMOTETomek']

def trained_SVM_model(word_vectors, mpaa_ratings, test_train, test_target):
  param_grid = ({ 'C':[0.1,1,100],'kernel':['rbf','poly','sigmoid','linear'],
                 'degree':[1,2,3],'gamma': [1, 0.1, 0.01, 0.001]})
  grid = GridSearchCV(SVC(), param_grid)

  model = grid.fit(word_vectors, mpaa_ratings)
  predict_test = model.predict(test_train)
  return (
      predict_test, 
      accuracy_score(test_target, predict_test), 
      f1_score(test_target, predict_test, average='weighted'),
      sum(1 for i, x in enumerate(test_target) if x == 3 and x == predict_test[i]) / sum(1 for i, x in enumerate(test_target) if x == 3)
    )

def trained_NB_model(word_vectors, mpaa_ratings, test_train, test_target):
  model = MultinomialNB().fit(word_vectors, mpaa_ratings)
  predict_test = model.predict(test_train)
  return (
      predict_test, 
      accuracy_score(test_target, predict_test), 
      f1_score(test_target, predict_test, average='weighted'),
      sum(1 for i, x in enumerate(test_target) if x == 3 and x == predict_test[i]) / sum(1 for i, x in enumerate(test_target) if x == 3)
    )

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
  return (
      predict_test, 
      accuracy_score(test_target, predict_test), 
      f1_score(test_target, predict_test, average='weighted'),
      sum(1 for i, x in enumerate(test_target) if x == 3 and x == predict_test[i]) / sum(1 for x in test_target if x == 3)
    )

def trained_Adaboost_model(word_vectors, mpaa_ratings, test_train, test_target, estimator = None):
    model = AdaBoostClassifier(base_estimator = estimator, n_estimators=100, random_state=42).fit(word_vectors, mpaa_ratings)
    predict_test = model.predict(test_train)
    return (
      predict_test, 
      accuracy_score(test_target, predict_test), 
      f1_score(test_target, predict_test, average='weighted'),
      sum(1 for i, x in enumerate(test_target) if x == 3 and x == predict_test[i]) / sum(1 for i, x in enumerate(test_target) if x == 3)
    )

# given data and a model, it runs the things, returns the things, and displays a confusian matrix if you ask nicely
def run_model(train_vector, train_labels, test_vector, test_labels, model, conf_matrix = None):
  if model == 'knn':
    output = trained_KNN_model(train_vector, train_labels.astype(int), test_vector, test_labels.astype(int))
  elif model == 'nb':
    output = trained_NB_model(train_vector, train_labels.astype(int), test_vector, test_labels.astype(int))
  elif model == 'svm':
    output = trained_SVM_model(train_vector, train_labels.astype(int), test_vector, test_labels.astype(int))
  elif model == 'adaboost':
    output = trained_Adaboost_model(train_vector, train_labels.astype(int), test_vector, test_labels.astype(int))
  elif model == 'adaboost_nb': 
    output = trained_Adaboost_model(train_vector, train_labels.astype(int), test_vector, test_labels.astype(int), estimator = MultinomialNB())

  predicted = output[0]
  # print(predicted)

  if not not conf_matrix:
    matrix = sklearn.metrics.confusion_matrix(test_labels, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["G", "PG", "Mature"])
    cm_display.plot()
    plt.rcParams.update({'font.size': 33})
    plt.title(conf_matrix)
    plt.show()
  return {'accuracy': output[1], 'f1': output[2], 'accuray for true mature': output[3]}

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

  #---Vectorizing Data---
  train_vector = data_preprocessing.word_vectorizer_train(type="train")
  train_labels = data_preprocessing.mpaa_pre_processing(type="train")
  test_vector = data_preprocessing.word_vectorizer_train(type="test")
  test_labels = data_preprocessing.mpaa_pre_processing(type="test")

  #---Resampled Data---
  x_rus, y_rus = imbalanced.RUS(train_vector, train_labels)
  x_tl, y_tl = imbalanced.TLinks(train_vector, train_labels)
  x_ros, y_ros = imbalanced.ROS(train_vector, train_labels)
  x_sm, y_sm = imbalanced.SMOTE_Reg(train_vector, train_labels)
  x_smt, y_smt = imbalanced.SMOTE_TL(train_vector, train_labels)

  all_data = [(train_vector, train_labels), (x_rus, y_rus), (x_tl, y_tl), (x_ros, y_ros), (x_sm, y_sm), (x_smt, y_smt)]

  x_en, y_en = imbalanced.ENN(train_vector, train_labels)

  # --- KNN ---
  #if you want a confusion matrix, set 'conf_matrix' param to data_set
  # for i, data_set in enumerate(DATA_SETS):
    # print(f"ur {data_set} knn accuracy is:", run_model(all_data[i][0], all_data[i][1], test_vector, test_labels, 'knn', conf_matrix = None))

  # --- Naive Bayes ---
  # print(f"ur enn nb accuracy is:", run_model(x_en, y_en, test_vector, test_labels, 'nb', conf_matrix = 'enn'))
  for i, data_set in enumerate(DATA_SETS):
    print(f"ur {data_set} nb accuracy is:", run_model(all_data[i][0], all_data[i][1], test_vector, test_labels, 'nb', conf_matrix = data_set))

  #--- SVM ---
  # for i, data_set in enumerate(DATA_SETS):
    # print(f"ur {data_set} svm accuracy is:", run_model(all_data[i][0], all_data[i][1], test_vector, test_labels, 'svm', conf_matrix = None))
  
  # --- Adaboost ---
  for i, data_set in enumerate(DATA_SETS):
    print(f"ur {data_set} ab accuracy is:", run_model(all_data[i][0], all_data[i][1], test_vector, test_labels, 'adaboost', conf_matrix = data_set))
  
  # --- Adaboost + NB ---
  for i, data_set in enumerate(DATA_SETS):
    print(f"ur {data_set} ab/nb accuracy is:", run_model(all_data[i][0], all_data[i][1], test_vector, test_labels, 'adaboost_nb', conf_matrix = data_set))
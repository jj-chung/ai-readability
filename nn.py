import numpy as np
from sklearn.metrics import mean_squared_error
import json
import data_preprocessing
from raw_data import *
from data_preprocessing import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

def train_nn(train_v, train_s, test_v, test_s):
    mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
    mlp.fit(train_v, train_s)

    predict_test = mlp.predict(test_v)
    accuracy = np.mean(predict_test == test_s)*100
    return accuracy

'''def create_plot(X, y, y_pred):
    plt.scatter(X, y, color="black")
    plt.plot(X, y_pred, color="blue", linewidth=2)
    plt.title("BT-easiness versus CEFR score")

    plt.xticks(())
    plt.yticks(())

    plt.show()'''


if __name__ == "__main__":
    #---Using TDIF Vectorization---
    train_vector = data_preprocessing.word_vectorizer_train(type="train")
    train_score = bt_easiness_train_data()
    test_vector = data_preprocessing.word_vectorizer_train(type="test")
    test_score = bt_easiness_test_data()
    accuracy = train_nn(train_vector, train_score, test_vector, test_score)


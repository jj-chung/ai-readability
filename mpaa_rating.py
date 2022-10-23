import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def get_KNN_clusters():
  neigh = NearestNeighbors(n_neighbors=10)
  neigh.fit(word_vectors)

if __name__ == "__main__":
  print(mpaa_ratings)
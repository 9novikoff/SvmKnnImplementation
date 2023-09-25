import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
  def __init__(self, k) -> None:
    self.k = k


  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train


  def count_distance(self, x, y):
    return np.sqrt(sum([(x[i] - y[i])**2 for i in range(len(x))]))
  
  
  def single_predict(self, item):
    most_frequent_y = [self.y_train._values[x] for x in np.argpartition([self.count_distance(item, i) for i in self.X_train._values], self.k)[0:3]]
    return max(most_frequent_y, key=lambda x: most_frequent_y.count(x))


  def predict(self, X_test):
    return [self.single_predict(x) for x in X_test._values]

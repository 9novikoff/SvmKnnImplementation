import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from knnm_implementation import *
from svm_implementation import *
from sklearn import svm

df = pd.read_csv('IRIS.csv')

X = df[df.columns[0: 4]]
y = df[df.columns[4]]

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, shuffle = True)

# knn = KNeighborsClassifier(n_neighbors=3)
  
# knn.fit(X_train, y_train)
  
# print(knn.predict(X_test))

# knnm = KNNClassifier(3)
# knnm.fit(X_train, y_train)

# print(knnm.predict(X_test))


svm = svm.SVC().fit(X_train, y_train)
print(svm.predict(X_test))

implemented_svm = SVM(set(y_train._values))
implemented_svm.fit(X_train._values, y_train._values)

print(implemented_svm.predict(X_test._values))

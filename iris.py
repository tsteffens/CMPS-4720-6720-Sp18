import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score

#read data in from file using pandas and then convert into numpy
data = pd.read_csv('iris.csv', header = -1)
data = pd.DataFrame.as_matrix(data)

#seperate data from classifications
X = data[:,0:3]
y = data[:,4]

#run train test split 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


#call SVM machine learning algorithm on data
svmclf =svm.SVC()
svmclf.fit(X_train, y_train)
svmpredictionTest=svmclf.predict(X_test)

#generate classification report for svm classifier precision recall and f1-score
print("#########################SVM Classification Report#####################################")
print(classification_report(y_test, svmpredictionTest))






#call KNN machine learning algorithm on data
knnclf =neighbors.KNeighborsClassifier()
knnclf.fit(X_train, y_train)
knnpredictionTest=knnclf.predict(X_test)

#generate classification report for svm classifier precision recall and f1-score
print("#########################knn Classification Report#####################################")
print(classification_report(y_test, knnpredictionTest))

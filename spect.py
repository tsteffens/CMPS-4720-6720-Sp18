import numpy as np
import pandas as pd
import math


#read train data in from file using pandas and then convert into numpy
train = pd.read_csv('SPECT.train.txt', header = -1)
train = pd.DataFrame.as_matrix(train)

#read test data in from file
test = pd.read_csv('SPECT.test.txt', header = -1)
test = pd.DataFrame.as_matrix(test)

print(train)
print(test)

#function to evaluate distance between instances
def euclidDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += pow((data1[x+1]-data2[x+1]),2)
    return math.sqrt(distance)

#find the k nearest instances
def getNearest(train,test,k):
    distances = []
    length = len(test)-1
    for i in range(len(train)):
        dist = euclidDistance(test, train[i], length)
        distances.append(dist)
    distances = np.asarray(distances)
    #print(distances)
    neighbors = distances.argsort()[:k]
    #print(neighbors)
    return neighbors

#vote to classify based on k nearest instances
def vote(neighbors):
    tally = 0
    for i in range(len(neighbors)):
        if (train[neighbors[i]][0] == 1):
            tally += 1
        else:
            tally -=1
    if (tally >= 0):
        return 1
    else:
        return 0
    
#evaluate Accuracy
def Accuracy(test, predictions):
    numRight = 0
    for i in range(len(test)):
        if test[i][0] == predictions[i]:
            numRight += 1
    return(numRight/float(len(test)))*100

#call functions
def main():
    predictions = []
    #########Set K##########
    k=2
    ########################
    for i in range(len(test)):
        neighbors = getNearest(train,test[i],k)
        classif = vote(neighbors)
        predictions.append(classif)
        print('Predicted: ', classif,'     Actual: ',test[i][0])
    accuracy = Accuracy(test, predictions)
    print('Accuracy: ', accuracy, '%')

main()

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import numpy as np
import argparse
import sys
import os
import csv

    
def analyze(classId,classifier,X_train,X_test,y_train,y_test,labels):
    classifier.fit(X_train,y_train)
    prediction = classifier.predict(X_test)
    correct = np.sum(y_test==prediction)/prediction.shape[0]
    confusion = confusion_matrix(y_test,prediction,labels=labels)

    acc = accuracy(confusion)
    rec = recall(confusion)
    pre = precision(confusion)
    toWrite = []
    toWrite.append(classId)
    toWrite.append(acc)
    for r in rec:
        toWrite.append(r)
    for p in pre:
        toWrite.append(p)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            toWrite.append(confusion[i,j])

    rows = [] 
    if os.path.isfile('a1_3.1.csv'):
        with open('a1_3.1.csv', 'r') as csvFile:
            classifierReader = csv.reader(csvFile,delimiter=',')
            for row in classifierReader:
                rows.append(row)
    with open('a1_3.1.csv','w') as csvFile:
        classifierWriter = csv.writer(csvFile,delimiter=',')
        for row in rows:
            classifierWriter.writerow(row)
        classifierWriter.writerow(toWrite)
    return correct

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    confusionNum = 0.0
    confusionDen = np.sum(C)
    for i in range(C.shape[0]):
        confusionNum += C[i,i]
    accuracy = 0.0
    if confusionDen > 0.0:
        accuracy = confusionNum/confusionDen
    return accuracy

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    R = []
    for k in range(4):
        denom = 0.0
        for j in range(4):
            denom += C[k,j]
        if denom > 0:
            R.append(C[k,k]/denom)
        else:
            R.append(0.0)
    return R

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    P = []
    for k in range(4):
        denom = 0.0
        for j in range(4):
            denom += C[j,k]
        if denom > 0:
            P.append(C[k,k]/denom)
        else:
            P.append(0.0)   
    return P

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    if os.path.isfile('a1_3.1.csv'):
        os.remove('a1_3.1.csv')
    features = np.load(filename)['arr_0']
    splitFeatures = train_test_split(features, test_size=0.2)
    train = splitFeatures[0]
    test = splitFeatures[1]

    X_train = train[:,:173]
    X_test = test[:,:173]
    y_train = train[:,173]
    y_test = test[:,173]
    labels = np.array([0,1,2,3])
    iBest = 0
    correct = []
    svcLinear = SVC(kernel='linear')
    correct.append(analyze(1,svcLinear,X_train,X_test,y_train,y_test,labels))
    print("Done svcLinear")
    
    svcRBF = SVC(kernel='rbf',gamma=2)
    correct.append(analyze(2,svcRBF,X_train,X_test,y_train,y_test,labels))
    print("Done svcRBF")

    rfc = RandomForestClassifier(n_estimators=10,max_depth=5)
    correct.append(analyze(3,rfc,X_train,X_test,y_train,y_test,labels))
    print("Done rfc")

    mlp = MLPClassifier(alpha=0.05)
    correct.append(analyze(4,mlp,X_train,X_test,y_train,y_test,labels))
    print("Done mlp")

    ada = AdaBoostClassifier()
    correct.append(analyze(5,ada,X_train,X_test,y_train,y_test,labels))
    print("Done ada")

    iBest = correct.index(max(correct))

    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    amounts = [1,5,10,15,20]
    print(X_train.shape)
    for i in amounts:
        print(i)
    print("Hello", iBest)
    return None
    #return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    # TODO : complete each classification experiment, in sequence.
    params = class31(args.input)
    class32(params[0],params[1],params[2],params[3],params[4])

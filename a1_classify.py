from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from scipy import stats

import numpy as np
import argparse
import sys
import os
import csv

def csvReadWriter(csvFilename, toWrite):
    rows = []
    if os.path.isfile(csvFilename):
        with open(csvFilename, 'r') as csvFile:
            csvReader = csv.reader(csvFile,delimiter=',')
            for row in csvReader:
                rows.append(row)
    with open(csvFilename,'w') as csvFile:
        csvWriter = csv.writer(csvFile,delimiter=',')
        for row in rows:
            csvWriter.writerow(row)
        csvWriter.writerow(toWrite)

def analyze(classId,classifier,X_train,X_test,y_train,y_test,labels):
    if classId != 5:
        return 0
    classifier.fit(X_train,y_train)
    prediction = classifier.predict(X_test)
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

    csvReadWriter('a1_3.1.csv',toWrite)
    return acc

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
    accs = []
    models = [(LinearSVC(),1), (SVC(kernel='rbf',gamma=2),2), (RandomForestClassifier(n_estimators=10,max_depth=5),3), (MLPClassifier(alpha=0.05),4), (AdaBoostClassifier(),5)]
    for model in models:
        accs.append(analyze(model[1],model[0],X_train,X_test,y_train,y_test,labels))
    iBest = accs.index(max(accs))

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
    if os.path.isfile('a1_3.2.csv'):
        os.remove('a1_3.2.csv')
    amounts = [1,5,10,15,20]
    labels = [0,1,2,3]
    accs = []
    if iBest == 0:
        model = LinearSVC()
    elif iBest == 1:
        model = SVC(kernel='rbf',gamma=2)
    elif iBest == 2:
        model = RandomForestClassifier(n_estimators=10,max_depth=5)
    elif iBest == 3:
        model = MLPClassifier(alpha=0.05)
    elif iBest == 4:
        model = AdaBoostClassifier()
    for i in amounts:
        X_k = X_train[:i*1000,:]
        y_k = y_train[:i*1000]
        model.fit(X_k,y_k)
        prediction = model.predict(X_test)
        confusion = confusion_matrix(y_test,prediction,labels=labels)
        accs.append(accuracy(confusion))
    X_1k = X_train[:1000,:]
    y_1k = y_train[:1000]
    csvReadWriter('a1_3.2.csv',accs)
    return (X_1k, y_1k)

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
    if os.path.isfile('a1_3.3.csv'):
        os.remove('a1_3.3.csv')
    numFeatures = [5,10,20,30,40,50]
    labels = [0,1,2,3]
    x = [X_1k,X_train]
    y = [y_1k,y_train]
    for k in numFeatures:
        selector = SelectKBest(f_classif, k)
        for ind in range(2):
            X_new = selector.fit_transform(x[ind], y[ind])
            pp = selector.pvalues_
            if ind == 1:
                csvReadWriter('a1_3.3.csv',[k]+pp.tolist())
    if i == 0:
        model = LinearSVC()
    elif i == 1:
        model = SVC(kernel='rbf',gamma=2)
    elif i == 2:
        model = RandomForestClassifier(n_estimators=10,max_depth=5)
    elif i == 3:
        model = MLPClassifier(alpha=0.05)
    elif i == 4:
        model = AdaBoostClassifier()
    accs = []
    for ind in range(2):
        selector = SelectKBest(f_classif, 5)
        selector = selector.fit(x[ind], y[ind])
        X_train_reduced = selector.transform(x[ind])
        X_test_reduced = selector.transform(X_test)
        model.fit(X_train_reduced,y[ind])
        prediction = model.predict(X_test_reduced)
        confusion = confusion_matrix(y_test,prediction,labels=labels)
        accs.append(accuracy(confusion))
    csvReadWriter('a1_3.3.csv',accs)


def class34( filename, i ):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    if os.path.isfile('a1_3.4.csv'):
        os.remove('a1_3.4.csv')
    models = [LinearSVC(), SVC(kernel='rbf',gamma=2), RandomForestClassifier(n_estimators=10,max_depth=5), MLPClassifier(alpha=0.05), AdaBoostClassifier()]
    features = np.load(filename)['arr_0']
    X = features[:,:173]
    y = features[:,173]
    labels = np.array([0,1,2,3])
    kf = KFold(n_splits = 5, shuffle=True)
    accs = np.zeros((5,5))
    fold = 0
    for train_index, test_index in kf.split(X):
        for m in range(len(models)):
            model = models[m]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train,y_train)
            prediction = model.predict(X_test)
            confusion = confusion_matrix(y_test,prediction,labels=labels)
            accs[fold,m] = accuracy(confusion)

        csvReadWriter('a1_3.4.csv',accs[fold,:].tolist())
        fold+=1
    a = accs[:,i]
    Ps = []
    for n in range(4):
        b = np.delete(accs,i,1)[:,n]
        S = stats.ttest_rel(a,b)
        Ps.append(S.pvalue)
    csvReadWriter('a1_3.4.csv',Ps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    params31 = class31(args.input)
    params32 = class32(params31[0],params31[1],params31[2],params31[3],params31[4])
    class33(params31[0],params31[1],params31[2],params31[3],params31[4],params32[0],params32[1])
    class34(args.input,params31[4])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import timeit
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from Testing import *


def normalizeData(x):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

'''
@ calculates accuracy using confusion matrix
'''
def Accuracy(y_true,y_pred):

    matrix = ConfusionMatrix(y_true, y_pred)
    return np.trace(matrix)/y_true.shape[0]

'''
@ calculates Recall using confusion matrix
'''
def Recall(y_true,y_pred):
    matrix = ConfusionMatrix(y_true, y_pred)
    recall = np.sum( matrix.diagonal() / np.sum(matrix, axis=1))
    return recall/matrix.shape[1]

'''
@ calculates Precision using confusion matrix
'''
def Precision(y_true,y_pred):
    matrix = ConfusionMatrix(y_true, y_pred)
    precision = np.sum( matrix.diagonal() / np.sum(matrix, axis=0))
    return precision/matrix.shape[0]

'''
@ Calculates Confusion matrix using numpy
@ (y_actual*numberOfClasses + y_test).reshape(numberOfClasses, numberOfClasses)
'''
def ConfusionMatrix(y_true,y_pred):
    unique = np.unique(y_true)
    n = len(unique)
    temp = (y_true-unique.min())*n + (y_pred-unique.min())
    hist, bin_edges = np.histogram(temp, bins=np.arange(0,n*n+1))
    return hist.reshape(n,n)

def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """

def KNN(X_train, X_test, Y_train):
    """
   :type X_train: numpy.ndarray
   :type X_test: numpy.ndarray
   :type Y_train: numpy.ndarray

   :rtype: numpy.ndarray
   """

def RandomForest(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: numpy.ndarray
    """

def PCA(X_train, N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """

def Kmeans(X_train, N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """

def SklearnSupervisedLearning(x_train,y_train,x_test):
    '''
    @ Logistic regression
    '''
    predictions = []
    startLogistic = timeit.default_timer()
    logisticReg = LogisticRegression(max_iter=400, n_jobs=5)
    logisticReg.fit(x_train, y_train)
    predictions.append(np.asarray(logisticReg.predict(x_test)))
    endLogistic = timeit.default_timer()
    print("Time take Logistic regression" ,   endLogistic - startLogistic)



    '''
    #@ SVM
    '''
    startSVM = timeit.default_timer()
    svmClasifier = svm.SVC(kernel='linear')
    svmClasifier.fit(x_train, y_train)
    predictions.append(np.asarray(svmClasifier.predict(x_test)))
    endSVM = timeit.default_timer()
    print("Time take SVM" ,  endSVM- startSVM )

    '''
    #@ Decision Trees
    '''
    startDT = timeit.default_timer()
    dt = DecisionTreeClassifier()
    dt = dt.fit(x_train,y_train)
    predictions.append(np.asarray(dt.predict(x_test)))
    endDT = timeit.default_timer()
    print("Time take DT" ,  endDT- startDT )

    '''
    #@ KNN
    '''
    startKNN = timeit.default_timer()
    KNNClasifier = KNeighborsClassifier(n_neighbors = 5,algorithm='auto')
    KNNClasifier.fit(x_train,y_train)
    predictions.append(np.asarray(KNNClasifier.predict(x_test)))
    endKNN = timeit.default_timer()
    print("Time take KNN", endKNN - startKNN)
    endTime = timeit.default_timer()
    print("Total time ", endTime - startLogistic)

    return predictions


def SklearnVotingClassifier(X_train,Y_train,X_test):
    estimator = []
    startTime = timeit.default_timer()
    estimator.append(('LR', LogisticRegression(max_iter=400, n_jobs=5)))
    estimator.append(('SVC',svm.SVC(kernel='linear')))
    estimator.append(('DTC',DecisionTreeClassifier()))
    estimator.append(('KNN',KNeighborsClassifier(n_neighbors = 5)))
    hard = VotingClassifier(estimators=estimator, voting='hard')
    hard.fit(X_train, Y_train)
    y_pred = hard.predict(X_test)
    endTime = timeit.default_timer()
    print("Total time ", endTime - startTime)
    return np.asarray(y_pred)

if __name__ == '__main__':
    # @transform the data
    data = pd.read_csv(r"/home/ravi/Desktop/DIC/Assignment1/data.csv")
    x_actual = normalizeData(data.iloc[:, :-1])
    y_actual = data.iloc[:, -1]

    # @ split the data set
    x_train, x_test, y_train, y_test = train_test_split(x_actual, y_actual, test_size=0.20)
    predicitons = SklearnSupervisedLearning(x_train, y_train, x_test)

    test = Testing(predicitons, x_test,y_test)
    test.run()
    SklearnVotingClassifier(x_train, y_train, x_test)
    testVoting = Testing([SklearnVotingClassifier(x_train, y_train, x_test)], x_test,y_test)
    testVoting.run()



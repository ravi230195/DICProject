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
from sklearn.model_selection import GridSearchCV
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

def gridSearch(X_train, Y_train, model, tuned_parameters):
    print("# Tuning hyper-parameters for {}".format(model.__class__))
    print()
    modelTuned = GridSearchCV(model, tuned_parameters, n_jobs=3)
    modelTuned.fit(X_train, Y_train)
    print(modelTuned.best_estimator_)
    print(modelTuned.best_params_)
    print(modelTuned.best_score_)
    means = modelTuned.cv_results_['mean_test_score']
    stds = modelTuned.cv_results_['std_test_score']
    result =[]
    for mean, std, params in zip(means, stds, modelTuned.cv_results_['params']):
        result.append([mean, params])
        print("%0.9f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return modelTuned, result



def printGridPlot(l,parameter, model, basedOn, xpara, axis, onParam):
    import matplotlib.pyplot as plt
    y = []
    x = []
    x1 = []
    y1 =[]
    label = ""
    label1 =""
    for i in l:
        if i[1][basedOn] == parameter[0]:
            y.append(i[0])
            x.append(i[1][xpara])
            label = basedOn+": " + i[1][basedOn]
        else:
            y1.append(i[0])
            x1.append(i[1][xpara])
            label1 = basedOn+ ": "  + i[1][basedOn]
    print(x)
    print(y)
    print(label)
    plt.plot(x, y, label=label)
    print(x1)
    print(y1)
    plt.plot(x1, y1, label=label1)
    plt.plot(x1, y1, label= onParam)
    plt.title(model)
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([min(min(y1), min(y)) - 0.0008, 1.00001])
    plt.show()

if __name__ == '__main__':
    # @transform the data
    data = pd.read_csv(r"/home/ravi/Desktop/DIC/Assignment1/data.csv")
    x_actual = normalizeData(data.iloc[:, :-1])
    y_actual = data.iloc[:, -1]

    # @ split the data set
    x_train, x_test, y_train, y_test = train_test_split(x_actual, y_actual, test_size=0.20)
    
    predicitons = SklearnSupervisedLearning(x_train, y_train, x_test)
    test = Testing(predicitons, x_test,y_test, True)
    test.run()

    SklearnVotingClassifier(x_train, y_train, x_test)
    testVoting = Testing([SklearnVotingClassifier(x_train, y_train, x_test)], x_test,y_test, True)
    testVoting.run()

   

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3],'C': [1, 10, 100,500, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100,500, 1000]}]
    modelTuned, SVM = gridSearch(x_train,y_train, svm.SVC(), tuned_parameters)
    print(SVM)
    print()

    #SVM =[[0.9894396064623441, {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}], [0.999175926655707, {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}], [0.9995116600678937, {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}], [0.9995727054707674, {'C': 500, 'gamma': 0.001, 'kernel': 'rbf'}], [0.9995727054707674, {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}], [0.9995726961544145, {'C': 1, 'kernel': 'linear'}], [0.9996642619296366, {'C': 10, 'kernel': 'linear'}], [0.9996642619296366, {'C': 100, 'kernel': 'linear'}], [0.9996642619296366, {'C': 500, 'kernel': 'linear'}], [0.9996642619296366, {'C': 1000, 'kernel': 'linear'}]]
    printGridPlot(SVM, ['linear', 'rbf'], "SVM", 'kernel', 'C', ['Regularization parameter', 'Accuracy'], 'C: [1, 10, 100,500, 1000]')
 
    print("For KNN")
    tuned_parameters = {'n_neighbors': [3,5,4,6,7], 'weights': ['distance'], 'metric': ['euclidean', 'manhattan']}
    modelTuned, KNNValues = gridSearch(x_train, y_train, KNeighborsClassifier(), tuned_parameters)
    print(KNNValues)
    
    #KNNValues = [[0.9044682952396977, {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}], [0.9116102999362576, {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'distance'}], [0.9095959367285478, {'metric': 'euclidean', 'n_neighbors': 4, 'weights': 'distance'}], [0.9151202033126342, {'metric': 'euclidean', 'n_neighbors': 6, 'weights': 'distance'}], [0.9153949379037126, {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'distance'}], [0.9898058555887032, {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}], [0.9913929848234748, {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}], [0.9909046542077217, {'metric': 'manhattan', 'n_neighbors': 4, 'weights': 'distance'}], [0.992919082629902, {'metric': 'manhattan', 'n_neighbors': 6, 'weights': 'distance'}], [0.9931632269759845, {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}]]
    printGridPlot(KNNValues, ['euclidean', 'manhattan'], "KNN", 'metric', 'n_neighbors', ['n_neighbors', 'Accuracy'], 'n_neighbors: [3,5,4,6,7]')

    print()
    print()
    print("Decision Trees")
    tuned_parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best'], 'min_samples_split': [2,3,5,7]}
    modelTuned, DT = gridSearch(x_train, y_train, DecisionTreeClassifier(), tuned_parameters)
    printGridPlot(DT, ['gini', 'entropy'], "DT", 'criterion', 'min_samples_split', ['min_samples_split', 'Accuracy'], 'min_samples_split: [2,3,5,7]')




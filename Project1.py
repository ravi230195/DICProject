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
from collections import Counter
#from Testing import *
import time


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
    '''Clusters contains the list of centroids of each cluster and the training data append to it'''
    k = len(Clusters)
    distance=0
    for i in range (1,k):
        distance+=np.sum(np.square(np.asarray(Clusters[i-1]) - Clusters[k-1][Clusters[k-1][:, -1] == i][:,:-1]))
    return distance

def KNN(X_train, X_test, Y_train):
    num_test=X_test.shape[0]
    num_train = X_train.shape[0]
    distance = np.zeros((num_test,num_train))
    distance = np.sqrt(np.sum(X_test**2, axis=1).reshape(num_test, 1) + np.sum(X_train**2, axis=1) - 2 * X_test.dot(X_train.T))
    newarr = np.argsort(distance,kind='heapsort',axis=1)  #indexes of the sorted rows 
    Y_train=np.asarray(Y_train)
    Y_test = Y_train[np.asarray(newarr)][:,:3]
    np.asarray(Y_test)
    count_list=[]
    for i in range(Y_test.shape[0]):
        count = Counter(Y_test[i])
        count.most_common(1)
        count_list.append(list(count.keys())[0])
    Y_test = np.asarray(count_list)
    return Y_test

'''
#@ Random Forest
'''
def isPure(x_train):
    unique_values, unique_counts = np.unique(x_train[:,-1], return_counts = True)
    #print(unique_values)
    #print("----------")
    #print(unique_counts)
    num_classes = len(unique_values)
    #print(num_classes)
    if num_classes == 1:
        return True
    else:
        return False

def classify(x_train):
    unique_values, unique_indices, unique_counts = np.unique(x_train[:,-1], return_index = True, return_counts = True)
    #print(unique_values)
    #print("--------------")
    #print(unique_indices)
    #print("--------------")
    #print(unique_counts)
    to_split_on = unique_counts.argmax()
    #print(to_split_on)
    return int(unique_values[to_split_on])

def total_possible_splits(x_train):
    possible_splits = {}
    for col in range(0, len(x_train[0,:])-1):
        possible_splits[col] = []
        unique_values = np.unique(x_train[:,col])
        #unique_values = np.unique(np.sort(x_train[:,col]))
        for i in range(len(unique_values)):
            if i != 0:
                v1 = unique_values[i]
                v0 = unique_values[i-1]
                split = (v0 + v1)/2
                possible_splits[col].append(split)
    return possible_splits

def split(x_train, split_on_index, split_on_value):
    left_split = []
    right_split = []
    for row in x_train:
        if row[split_on_index] < split_on_value:
            left_split.append(row)
        else:
            right_split.append(row)
    left_split = np.asarray(left_split)
    right_split = np.asarray(right_split)
    return left_split, right_split

def gini_index(x_train):
    unique_values, unique_indices, unique_counts = np.unique(x_train[:,-1], return_index = True, return_counts = True)
    num_classes = len(unique_values)
    p1 = p2 = 0
    for i in unique_counts:
        p1 = i * i
        p2 = p2 + p1

    prob1 = p2/(len(x_train[:,-1])**2)
    gini_index = 1 - prob1
    #print(gini_index)
    return gini_index

def gini_index_after_split(x_train, left_split, right_split):
    n = len(x_train)
    left_split_prob = len(left_split) / n
    right_split_prob = len(right_split) / n

    gini =  (left_split_prob * gini_index(left_split)) + (right_split_prob * gini_index(right_split))
    return gini

def best_split(x_train, possible_splits):
    gini = 2
    for col in possible_splits:
        for i in possible_splits[col]:
            left_split, right_split = split(x_train, col, i)
            current_gini = gini_index_after_split(x_train, left_split, right_split)
            if current_gini <= gini:
                gini = current_gini
                best_split_col = col
                best_split_val = i
    return best_split_col, best_split_val

def DT(x_train, counter):
    #treenode = Tree()
    if isPure(x_train):
        classification = classify(x_train)
        return classification
    else:
        counter += 1
        possible_splits = total_possible_splits(x_train)
        best_split_col, best_split_val = best_split(x_train, possible_splits)
        left_split, right_split = split(x_train, best_split_col, best_split_val)
        question = str(best_split_col) + ":" + str(best_split_val)
        sub_tree = {question: []}
        less_than = DT(left_split, counter)
        greater_than_equal = DT(right_split, counter)
        sub_tree[question].append(less_than)
        sub_tree[question].append(greater_than_equal)
        return sub_tree

def bootstrap_data(x_train, num_of_bootstrap):
    np.random.seed(15)
    bootstrap_ind = np.random.randint(low = 0, high = len(x_train), size = num_of_bootstrap)
    x_bootstrapped = x_train[bootstrap_ind]
    return x_bootstrapped

def random_forest(x_train, num_of_trees, num_of_bootstrap):
    forest = []
    for i in range(num_of_trees):
        x_bootstrapped = bootstrap_data(x_train, num_of_bootstrap)
        tree = DT(x_bootstrapped, 0)
        forest.append(tree)

    return forest

def prediction(x_test, decision_tree):
    question = list(decision_tree.keys())[0]
    split_col, split_val = question.split(":")
    #print(split_col)

    if str(x_test[int(split_col)]) < split_val:
        answer = decision_tree[question][0]
    else:
        answer = decision_tree[question][1]

    remainder = answer
    if not isinstance(answer, dict):
        return answer
    return prediction(x_test, remainder)

def decision_tree_prediction(x_test, decision_tree):
    predictions = []
    for i in range(0, len(x_test)):
        predictions.append(prediction(x_test[i], decision_tree))
    return predictions


def random_forest_prediction(x_test, forest):
    rf_predictions = {}
    for i in range(len(forest)):
        prediction = decision_tree_prediction(x_test, forest[i])
        #rf_predictions.append(prediction)
        rf_predictions[i] = prediction

    return rf_predictions

def RandomForest(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: numpy.ndarray
    """
    forest = random_forest(x_train, 11, 100)
    print(time.ctime(time.time()))
    rf_predictions = random_forest_prediction(x_test, forest)
    print(time.ctime(time.time()))
    df_predictions = pd.DataFrame(rf_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    #print(Accuracy(y_test, random_forest_predictions))
    return random_forest_predictions.to_numpy()

def PCA(X_train, N):
    M=np.mean(X_train.T, axis=1)
    C= X_train - M
    V = np.cov(C.T)
    values, vectors = np.linalg.eig(V)
    values=values.argsort()[::-1][:N]
    vectors = vectors[: ,values]
    return vectors

def Kmeans(X_train, N):
    num_training=X_train.shape[0]
    num_features=X_train.shape[1]
    iterations = 100
    
    mean = np.mean(X_train,axis=0)
    standard_deviation = np.std(X_train,axis=0)
    centroids= (np.random.randn(N,num_features)*standard_deviation) + mean
    centroids_old = np.zeros(centroids.shape)
    centroids_new = np.copy(centroids)
    distance = np.zeros((num_training,N))
    clusters = np.zeros(num_training)
    diff = np.linalg.norm(np.square(centroids_old - centroids_new))
    for j in range(iterations):
        for i in range(N):
            distance[:,i] = np.linalg.norm(X_train-centroids[i],axis=1)
        clusters = np.argmin(distance, axis = 1)
        centroids_old = np.copy(centroids_new)
        
        for i in range(N):
            centroids_new[i] = np.mean(X_train[clusters == i], axis = 0)
        diff = np.linalg.norm(np.square(centroids_new - centroids_old))
        #print(diff)
        if(diff==0):
            break;
    #WCSS
    clus = list(centroids)
    clus.append(np.append(x_train,clusters.reshape(clusters.shape[0],1), axis = 1))
    print("WCSS:", WCSS(clus))
    return centroids_new

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
    data = pd.read_csv(r'C:/Users/akula/Documents/DIC/Assignment1/data.csv')
    x_actual = normalizeData(data.iloc[:, :-1])
    y_actual = data.iloc[:, -1]

    # @ split the data set
    x_train, x_test, y_train, y_test = train_test_split(x_actual, y_actual, test_size=0.20)

    # @Part 1
    # @KNN
    print("Accuracy for KNN Part-1 with 3 nearest neighbors:" ,round(Accuracy(y_test,KNN(x_train,x_test,y_train)),3)*100,"\n")
    
    # @Kmeans
    centroids = Kmeans(x_train,11)
    print("\nCluster Centroids after Kmeans: \n", centroids)
    
    #PCA
    print("\nPCA with 5 features: \n", PCA(x_train,5))
    
    # @RandomForest
 
    random_forest_predictions = RandomForest(x_train, y_train, x_test)
    print("Accuracy of Random Forest: " + Accuracy(y_test, random_forest_predictions))


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
    
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
import random
from Testing import *
import time


class RandomForestClassifier:
    '''
    #@ Random Forest
    '''

    def isPure(self, x_train):
        unique_values, unique_counts = np.unique(x_train[:, -1], return_counts=True)
        num_classes = len(unique_values)
        if num_classes == 1:
            return True
        else:
            return False

    def classify(self, x_train):
        unique_values, unique_indices, unique_counts = np.unique(x_train[:, -1], return_index=True, return_counts=True)
        to_split_on = unique_counts.argmax()
        return int(unique_values[to_split_on])

    def total_possible_splits(self, x_train):
        possible_splits = {}
        for col in range(0, len(x_train[0, :]) - 1):
            possible_splits[col] = []
            possible_splits[col].append(np.median(x_train[:, col]))
        return possible_splits

    def split(self, x_train, split_on_index, split_on_value):
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

    def gini_index(self, x_train):
        if len(x_train) > 0:
            unique_values, unique_indices, unique_counts = np.unique(x_train[:, -1], return_index=True,
                                                                     return_counts=True)
            p1 = 0
            p2 = 0
            for i in unique_counts:
                p1 = i * i
                p2 = p2 + p1

            prob1 = p2 / (len(x_train[:, -1]) ** 2)
            gini_index = 1 - prob1
            return gini_index
        else:
            return 0

    def gini_index_after_split(self, x_train, left_split, right_split):
        n = len(x_train)
        left_split_prob = len(left_split) / n
        right_split_prob = len(right_split) / n
        gini = (left_split_prob * self.gini_index(left_split)) + (right_split_prob * self.gini_index(right_split))
        return gini

    def best_split(self, x_train, possible_splits):
        gini = self.gini_index(x_train)
        for col in possible_splits:
            for i in possible_splits[col]:
                left_split, right_split = self.split(x_train, col, i)
                if gini != 0:
                    current_gini = self.gini_index_after_split(x_train, left_split, right_split)
                if current_gini <= gini:
                    gini = current_gini
                    best_split_col = col
                    best_split_val = i
        return best_split_col, best_split_val

    def DT(self, x_train, counter):
        if self.isPure(x_train):
            classification = self.classify(x_train)
            return classification
        else:
            counter += 1
            possible_splits = self.total_possible_splits(x_train)
            best_split_col, best_split_val = self.best_split(x_train, possible_splits)
            left_split, right_split = self.split(x_train, best_split_col, best_split_val)
            if len(left_split) == 0 or len(right_split) == 0:
                classification = self.classify(x_train)
                return classification
            question = str(best_split_col) + ":" + str(best_split_val)
            sub_tree = {question: []}
            less_than = self.DT(left_split, counter)
            greater_than_equal = self.DT(right_split, counter)
            sub_tree[question].append(less_than)
            sub_tree[question].append(greater_than_equal)
            return sub_tree

    def randominzer(self, x_train):
        print(x_train[0])
        print(x_train[:,1])
        l = [i for i in range(0,48)]
        print(l)
        sampling = random.choices(l, k=7)
        print(sampling)
        x = np.ndarray((x_train.shape[0], 7))
        for i in sampling:
            x = np.insert(x,0,  x_train[:,i], axis =1)
        print(x.shape)
        x = np.insert(x, 7, x_train[:,48], axis=1)
        return x

    def bootstrap_data(self, x_train, num_of_bootstrap):
        # np.random.seed
        bootstrap_ind = np.random.randint(low=0, high=len(x_train), size=num_of_bootstrap)
        x_bootstrapped = x_train[bootstrap_ind]
        return x_bootstrapped

    def random_forest(self, x_train, num_of_trees, num_of_bootstrap):
        forest = []
        for i in range(num_of_trees):
            x_bootstrapped = self.bootstrap_data(x_train, num_of_bootstrap)
            tree = self.DT(x_bootstrapped, 0)
            forest.append(tree)

        return forest

    def prediction(self, x_test, decision_tree):
        question = list(decision_tree.keys())[0]
        split_col, split_val = question.split(":")

        if str(x_test[int(split_col)]) < split_val:
            answer = decision_tree[question][0]
        else:
            answer = decision_tree[question][1]

        remainder = answer
        if not isinstance(answer, dict):
            return answer
        return self.prediction(x_test, remainder)

    def decision_tree_prediction(self, x_test, decision_tree):
        predictions = []
        for i in range(0, len(x_test)):
            predictions.append(self.prediction(x_test[i], decision_tree))
        return predictions

    def random_forest_prediction(self, x_test, forest):
        rf_predictions = {}
        for i in range(len(forest)):
            prediction = self.decision_tree_prediction(x_test, forest[i])
            rf_predictions[i] = prediction

        return rf_predictions



def Normalize(X_train, X_test):
    X_train = normalizeData(X_train)
    X_test = normalizeData(X_test)
    X_test1 = np.append(X_train, X_test, axis = 0)
    X_test1 = normalizeData(X_test1)
    num_train = X_train.shape[0]
    X_train = X_test1[0:num_train,:]
    X_test = X_test1[num_train:X_test1.shape[0],:]
    return X_train, X_test



def normalizeData(x):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(x)
    return scaled_data

'''
@ calculates accuracy using confusion matrix
'''


def Accuracy(y_true, y_pred):
    matrix = ConfusionMatrix(y_true, y_pred)
    return np.trace(matrix) / y_true.shape[0]


'''
@ calculates Recall using confusion matrix
'''


def Recall(y_true, y_pred):
    matrix = ConfusionMatrix(y_true, y_pred)
    recall = np.sum(matrix.diagonal() / np.sum(matrix, axis=1))
    return recall / matrix.shape[1]


'''
@ calculates Precision using confusion matrix
'''


def Precision(y_true, y_pred):
    matrix = ConfusionMatrix(y_true, y_pred)
    precision = np.sum(matrix.diagonal() / np.sum(matrix, axis=0))
    return precision / matrix.shape[0]


'''
@ Calculates Confusion matrix using numpy
@ (y_actual*numberOfClasses + y_test).reshape(numberOfClasses, numberOfClasses)
'''


def ConfusionMatrix(y_true, y_pred):
    unique = np.unique(y_true)
    n = len(unique)
    temp = (y_true - unique.min()) * n + (y_pred - unique.min())
    hist, bin_edges = np.histogram(temp, bins=np.arange(0, n * n + 1))
    return hist.reshape(n, n)


def WCSS(Clusters):
    k = len(Clusters)
    distance=0
    centroids=[[]]*k
    for i in range(0,k):
        centroids[i]=np.mean(Clusters[i],axis=0)
    for i in range(0,k):
        distance+=np.linalg.norm(np.square(centroids[i]-Clusters[i]))
    return distance


def KNN(X_train, X_test, Y_train):
    X_train, X_test = Normalize(X_train, X_test)
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



def RandomForest(x_train, y_train, x_test):
        """
        :type X_train: numpy.ndarray
        :type X_test: numpy.ndarray
        :type Y_train: numpy.ndarray
        :rtype: numpy.ndarray
        """
        x_train = np.insert(x_train, 48, y_train, axis=1)
        x_test = np.insert(x_test, 48, 0 ,axis=1)

        randomForest = RandomForestClassifier()
        forest = randomForest.random_forest(x_train, 15, 21550)
        rf_predictions = randomForest.random_forest_prediction(x_test, forest)
        df_predictions = pd.DataFrame(rf_predictions)
        random_forest_predictions = df_predictions.mode(axis=1)[0]
        return random_forest_predictions.to_numpy()


def PCA(X_train, N):
    M = np.mean(X_train.T, axis=1)
    C = X_train - M
    V = np.cov(C.T)  # 48x48 matrix
    values, vectors = np.linalg.eig(V)
    values = values.argsort()[::-1][:N]
    vectors = vectors[:, values]
    return vectors


def Kmeans(X_train, N):
    num_training = X_train.shape[0]
    num_features = X_train.shape[1]
    iterations = 100

    mean = np.mean(X_train, axis=0)
    standard_deviation = np.std(X_train, axis=0)
    centroids = (np.random.randn(N, num_features) * standard_deviation) + mean
    centroids_old = np.zeros(centroids.shape)
    centroids_new = np.copy(centroids)
    distance = np.zeros((num_training, N))
    cluster_index = np.zeros(num_training)
    diff = np.linalg.norm(np.square(centroids_old - centroids_new))
    for j in range(iterations):
        for i in range(N):
            distance[:, i] = np.linalg.norm(X_train - centroids_new[i], axis=1)
        cluster_index = np.argmin(distance, axis=1)  # Y_pred values for the train data
        centroids_old = np.copy(centroids_new)

        for i in range(N):
            centroids_new[i] = np.mean(X_train[cluster_index == i], axis=0)
        diff = np.linalg.norm(np.square(centroids_new - centroids_old))
        # print(diff)
        if (diff == 0):
            break;
    # X_train=np.append(X_train,cluster_index.reshape(cluster_index.shape[0],1), axis = 1)
    output = [[]] * N
    for i in range(N):
        output[i] = X_train[cluster_index == i][:, :-1]
    # print(output)
    WCSS(output)
    ''' Calling the WCSS Function '''
    return output

def SklearnSupervisedLearning(x_train, y_train, x_test):
    '''
    @ Logistic regression
    '''
    x_train, x_test = Normalize(x_train, x_test)
    predictions = []
    startLogistic = timeit.default_timer()
    logisticReg = LogisticRegression(max_iter=1000, n_jobs=5)
    logisticReg.fit(x_train, y_train)
    predictions.append(np.asarray(logisticReg.predict(x_test)))
    endLogistic = timeit.default_timer()
    #print("Time take Logistic regression", endLogistic - startLogistic)

    '''
    #@ SVM
    '''
    startSVM = timeit.default_timer()
    svmClasifier = svm.SVC(kernel='linear')
    svmClasifier.fit(x_train, y_train)
    predictions.append(np.asarray(svmClasifier.predict(x_test)))
    endSVM = timeit.default_timer()
    #print("Time take SVM", endSVM - startSVM)

    '''
    #@ Decision Trees
    '''
    startDT = timeit.default_timer()
    dt = DecisionTreeClassifier()
    dt = dt.fit(x_train, y_train)
    predictions.append(np.asarray(dt.predict(x_test)))
    endDT = timeit.default_timer()
    #print("Time take DT", endDT - startDT)

    '''
    #@ KNN
    '''
    startKNN = timeit.default_timer()
    KNNClasifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    KNNClasifier.fit(x_train, y_train)
    predictions.append(np.asarray(KNNClasifier.predict(x_test)))
    endKNN = timeit.default_timer()
    #print("Time take KNN", endKNN - startKNN)
    endTime = timeit.default_timer()
    #print("Total time ", endTime - startLogistic)

    return predictions


def SklearnVotingClassifier(X_train, Y_train, X_test):
    estimator = []
    startTime = timeit.default_timer()
    X_train, X_test = Normalize(X_train, X_test)
    estimator.append(('LR', LogisticRegression(max_iter=1000, n_jobs=5)))
    estimator.append(('SVC', svm.SVC(kernel='linear')))
    estimator.append(('DTC', DecisionTreeClassifier()))
    estimator.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
    hard = VotingClassifier(estimators=estimator, voting='hard')
    hard.fit(X_train, Y_train)
    y_pred = hard.predict(X_test)
    endTime = timeit.default_timer()
    #print("Total time ", endTime - startTime)
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
    result = []
    for mean, std, params in zip(means, stds, modelTuned.cv_results_['params']):
        result.append([mean, params])
        print("%0.9f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return modelTuned, result


def printGridPlot(l, parameter, model, basedOn, xpara, axis, onParam):
    import matplotlib.pyplot as plt
    y = []
    x = []
    x1 = []
    y1 = []
    label = ""
    label1 = ""
    for i in l:
        if i[1][basedOn] == parameter[0]:
            y.append(i[0])
            x.append(i[1][xpara])
            label = basedOn + ": " + i[1][basedOn]
        else:
            y1.append(i[0])
            x1.append(i[1][xpara])
            label1 = basedOn + ": " + i[1][basedOn]
    plt.plot(x, y, label=label)
    plt.plot(x1, y1, label=label1)
    plt.plot(x1, y1, label=onParam)
    plt.title(model)
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([min(min(y1), min(y)) - 0.0008, 1.00001])
    plt.show()




############## Used For Testing #####################################

if __name__ == '__main__':
    startTime  = timeit.default_timer()
    # @transform the data
    data = pd.read_csv(r"/home/ravi/Desktop/DIC/Assignment1/data.csv")
    x_unNormalized = data.iloc[:, :-1]
    #x_actual = normalizeData(data.iloc[:, :-1])
    y_actual = data.iloc[:, -1]
    #x_actual = x_actual
    y_actual = y_actual.to_numpy()

    x_unNormalized = x_unNormalized.to_numpy()
    y_unNormalized = y_actual
    # @ split the data set
    #x_train, x_test, y_train, y_test = train_test_split(x_actual, y_actual, test_size=0.20)

    # @ unnormailized
    x_train, x_test, y_train, y_test = train_test_split(x_unNormalized, y_unNormalized, test_size=0.20)


    # @Part 1
    # @KNN
    print("\n Accuracy for KNN Part-1 with 3 nearest neighbors:",
          round(Accuracy(y_test, KNN(x_train, x_test, y_train)), 3) * 100, "\n")

    # @Kmeans
    centroids = Kmeans(x_train, 11)
    #print("\nCluster Centroids after Kmeans: \n", centroids)

    # PCA
    pcaValues = PCA(x_train, 5)
    print("\nPCA with 5 features: \n")
 
    random_forest_predictions = RandomForest(x_train, y_train, x_test)
    print("Accuracy of Random Forest: ", Accuracy(y_test, random_forest_predictions))
    
    
    predicitons = SklearnSupervisedLearning(x_train, y_train, x_test)
    test = Testing(predicitons, x_test, y_test, True)
    test.run()

    SklearnVotingClassifier(x_train, y_train, x_test)
    testVoting = Testing([SklearnVotingClassifier(x_train, y_train, x_test)], x_test, y_test, True)
    testVoting.run()
    endTime = timeit.default_timer()
    print("Total Project time ", endTime-startTime )
      
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3], 'C': [1, 10, 100, 500, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 500, 1000]}]
    modelTuned, SVM = gridSearch(x_train, y_train, svm.SVC(), tuned_parameters)
    print(SVM)
    print()

    printGridPlot(SVM, ['linear', 'rbf'], "SVM", 'kernel', 'C', ['Regularization parameter', 'Accuracy'],
                  'C: [1, 10, 100,500, 1000]')

    print("For KNN")
    tuned_parameters = {'n_neighbors': [3, 5, 4, 6, 7], 'weights': ['distance'], 'metric': ['euclidean', 'manhattan']}
    modelTuned, KNNValues = gridSearch(x_train, y_train, KNeighborsClassifier(), tuned_parameters)
    print(KNNValues)

    printGridPlot(KNNValues, ['euclidean', 'manhattan'], "KNN", 'metric', 'n_neighbors', ['n_neighbors', 'Accuracy'],
                  'n_neighbors: [3,5,4,6,7]')

    print()
    print()
    print("Decision Trees")
    tuned_parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best'], 'min_samples_split': [2, 3, 5, 7]}
    modelTuned, DT = gridSearch(x_train, y_train, DecisionTreeClassifier(), tuned_parameters)
    printGridPlot(DT, ['gini', 'entropy'], "DT", 'criterion', 'min_samples_split', ['min_samples_split', 'Accuracy'],
                  'min_samples_split: [2,3,5,7]')




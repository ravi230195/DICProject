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
from Assignment import *

class Testing:
    def __init__(self, predicitons, x_test, y_test):
        self.predicitons = predicitons
        self.x_test = x_test
        self.y_test = y_test



    def plotModels(self,confusionMatrix):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        i=0
        for row in ax:
            for col in row:
                sns.heatmap(confusionMatrix[i], annot = True)
                i +=1
                col.plot()
        plt.show()

    def run(self):
        forPlot=[]
        for i in range(len(self.predicitons)):
            print('Model ', i+1)
            metrix = []
            metrix.append(format(metrics.accuracy_score(self.y_test, self.predicitons[i]), '.5f'))
            metrix.append(format(metrics.recall_score(self.y_test, self.predicitons[i], average='macro'), '.5f'))
            metrix.append(format(metrics.precision_score(self.y_test,self.predicitons[i],  average='macro'), '.5f'))

            calculateValues = [format(Accuracy(self.y_test, self.predicitons[i]), '.5f'), format(Recall(self.y_test, self.predicitons[i]), '.5f'), format(Precision(self.y_test, self.predicitons[i]), '.5f')]
            print(metrix)
            print(calculateValues)
            confusionSklearns = confusion_matrix(self.y_test, self.predicitons[i])
            confusionCalculated = ConfusionMatrix(self.y_test, self.predicitons[i])
            if (metrix == calculateValues and np.array_equal(confusionSklearns, confusionCalculated)):
                print("TestCase pass ")
                forPlot.append(confusionCalculated)
                sns.heatmap(confusionCalculated, annot=True)
                plt.show()
            else:
                print("Test Failed")

        #self.plotModels(forPlot)

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
from Predictive_analytics import *

class Testing:
    def __init__(self, predicitons, x_test, y_test, plot):
        self.predicitons = predicitons
        self.x_test = x_test
        self.y_test = y_test
        self.plot = plot



    def plotModels(self, grid):
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax = [ax1,ax2,ax3,ax4]
        color = [plt.cm.winter, plt.cm.autumn, plt.cm.summer, plt.cm.spring]
        lables = ['Logistic Regression', 'SVM', 'Decision Tree', 'KNN']
        fig1.tight_layout(pad=3.0)
        for k in range(len(grid)):
            print(grid[k])
            ax[k].imshow(grid[k], cmap = color[k], interpolation='none', aspect='auto')
            ax[k].set_title(lables[k])
            ax[k].set(xlabel='Actual', ylabel='Predicted')
            for (j, i), label in np.ndenumerate(grid[k]):
                ax[k].text(i, j, label, ha='center', va='center')

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
                #sns.heatmap(confusionCalculated, annot=True)
                #plt.show()
            else:
                print("Test Failed")
        #if self.plot == True:
        #    self.plotModels(forPlot)

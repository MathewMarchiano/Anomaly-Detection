from numpy import genfromtxt
from sklearn.preprocessing.imputation import Imputer
from sklearn import preprocessing
import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class Predictor():
    def __init__(self):
        pass

    def trainModel(self, data, labels, model):
        if model == 1:
            classifier = svm.SVC(gamma='auto')
        elif model == 2:
            classifier = DecisionTreeClassifier(random_state=0)
        elif model == 3:
            classifier = LinearDiscriminantAnalysis()
        elif model == 4:
            classifier = KNeighborsClassifier(n_neighbors=15)
        elif model == 5:
            classifier = LogisticRegression(random_state=1)
        elif model == 6:
            numKnownClasses = len(np.unique(labels))
            print("Number of Nodes:", numKnownClasses)
            classifier = MLPClassifier(random_state=1, hidden_layer_sizes=(numKnownClasses,))
        elif model == 7:
            classifier = GaussianNB()
        elif model == 8:
            classifier = RandomForestClassifier(random_state=0)
        else:
            print("Specify Classifier")
        classifier = classifier.fit(data, labels)

        return classifier

class DataManager:

    def __init__(self):
        pass

    def getData(self, labelsColumn, dataBeginIndex, dataEndIndex, dataset):
        importedDataset = pd.read_csv(dataset, header=None)
        numColumns = len(importedDataset.columns)
        dataValues = genfromtxt(dataset, delimiter=',', usecols=range(dataBeginIndex, dataEndIndex)).tolist()

        # 1 == labels are in the first column. -1 == labels are in the last column
        if (labelsColumn == 1):
            labels = importedDataset.ix[:, 0].tolist()
        elif (labelsColumn == -1):
            labels = importedDataset.ix[:, (numColumns - 1)].tolist()

        return dataValues, labels

    def preprocessData(self, data):
        imputer = Imputer(missing_values=np.nan, strategy='mean')
        imputer.fit(data)
        imputedData = imputer.transform(data)  # nan values will take on mean
        scaledData = preprocessing.scale(imputedData).tolist()

        return scaledData

    def getSmallClasses(self, data, labels):
        uniqueLabels = np.unique(labels)
        labelsToRemove = []
        dataToRemove = []
        indicesToRemove = []

        for uniqueLabel in uniqueLabels:
            indices = []
            index = 0
            for label in labels:
                if label == uniqueLabel:
                    indices.append(index)
                index += 1
            if (len(indices) < 3):
                for index in indices:
                    labelsToRemove.append(labels[index])
                    dataToRemove.append(data[index])
                    indicesToRemove.append(index)

        return indicesToRemove, dataToRemove, labelsToRemove

    def removeSmallClasses(self, data, labels, indicesToRemove):
        sortedIndicies = sorted(indicesToRemove, reverse=True)
        for index in sortedIndicies:
            del labels[index]
            del data[index]

        return data, labels
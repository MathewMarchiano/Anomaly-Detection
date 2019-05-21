import numpy as np
from numpy import genfromtxt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing.imputation import Imputer
from sklearn import preprocessing
import pandas as pd


class Predictor():

    dataset = ""

    def __init__(self, dataset):
        self.dataset = dataset

    def getData(self, labelsColumn, dataBeginIndex, dataEndIndex):
        importedDataset = pd.read_csv(self.dataset, header=None)
        numColumns = len(importedDataset.columns)
        dataValues = genfromtxt(self.dataset, delimiter=',', usecols=range(dataBeginIndex, dataEndIndex)).tolist()

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

    def trainModel(self, data, labels, model):
        if model == 1:
            classifier = svm.SVC(gamma='auto')
        elif model == 2:
            classifier = DecisionTreeClassifier(random_state=0)
        elif model == 3:
            classifier = LinearDiscriminantAnalysis()
        elif model == 4:
            classifier = KNeighborsClassifier(n_neighbors=2)
        else:
            print("Specify Classifier")
        classifier = classifier.fit(data, labels)

        return classifier



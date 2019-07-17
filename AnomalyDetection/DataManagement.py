import numpy as np
from sklearn import preprocessing
import pandas as pd
from numpy import genfromtxt
import random
from sklearn.preprocessing.imputation import Imputer


# Retrieves and processes the data from a given CSV file or web address. Also deals with relabeling original labels into
# labels dictated by the classifiers of a given codebook.
class DatasetHandler():

    def __init__(self, codebook):
        self.codebook = codebook

    # Gets data from given dataset.
    def getData(self, dataset, labelsColumn, dataBeginIndex, dataEndIndex):
        importedDataset = pd.read_csv(dataset, header=None)
        numColumns = len(importedDataset.columns)
        dataValues = genfromtxt(dataset, delimiter = ',', usecols = range(dataBeginIndex, dataEndIndex)).tolist()

        #1 == labels are in the first column. -1 == labels are in the last column
        if(labelsColumn == 1):
            labels = importedDataset.ix[:, 0].tolist()
        elif(labelsColumn == -1):
            labels = importedDataset.ix[:, (numColumns - 1)].tolist()

        return dataValues, labels

    # Preprocesses the data.
    def preprocessData(self, data):
        imputer = Imputer(missing_values = np.nan, strategy = 'mean')
        imputer.fit(data)
        imputedData = imputer.transform(data) #nan values will take on mean
        scaledData = preprocessing.scale(imputedData).tolist()

        return scaledData

    def assignCodeword(self, labels):
        labelDictionary = {}
        updatedLabels = []
        random.shuffle(self.codebook)  # In order to randomize assignment of codewords.

        for originalLabel in labels:
            labelDictionary[originalLabel] = -1
        for label, codeword in zip(labelDictionary, self.codebook):
            labelDictionary[label] = codeword

        #Updating every original label to its corresponding codeword
        for label in labels:
            updatedLabels.append(labelDictionary[label])

        return updatedLabels, labelDictionary

    # Binarize labels based off of the assigned codeword for each class.
    # Needed for training later.
    # I'm pretty sure this handles creating the binary columns that are passed
    # (each one) to a unique classifier for training.
    def binarizeLabels(self, labelDictionary):
        updatedLabelsList = []
        classifierList = []
        for label in labelDictionary:
            updatedLabelsList.append(labelDictionary[label])

        codewordBits = len(updatedLabelsList[0])
        numClasses = len(updatedLabelsList)
        classifierIndex = 0
        classifierNumber = 0
        count = 1
        #The number of indices in a classifier is equal to the number of codewords/classes.
        #The number of classifiers is equal to the length or total number of bits in a codeword.
        #A classifier is made by getting a particular index's value for each and every codeword in
        #a codebook for every element in the codewords.
        while(classifierNumber < codewordBits):
            tempClassifier = []
            while(classifierIndex < numClasses):
                count += 1

                tempClassifier.append(updatedLabelsList[classifierIndex][classifierNumber])
                classifierIndex += 1

            classifierIndex = 0
            classifierNumber += 1
            classifierList.append(tempClassifier)

        #Creating a dictionary of what all the original labels will be assumed to be using the classifiers.
        #Will use this later before training when updating all the labels to what their new binary value will be.
        tempDictionary = {}
        classifierDictionaryList = []
        for classifier in classifierList:
            for index, origLabel in zip(classifier, labelDictionary):
                tempDictionary[origLabel] = index

            classifierDictionaryList.append(tempDictionary)
            tempDictionary = {}

        return classifierDictionaryList

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

    def getHoldoutIndices(self, labels, labelsToRemove):
        uniqueLabels = np.unique(labels)
        uniqueLabelsToRemove = np.unique(labelsToRemove)
        index = 0
        holdoutIndices = []

        for label in uniqueLabels:
            if label not in uniqueLabelsToRemove:
                holdoutIndices.append(index)
            index += 1

        return holdoutIndices


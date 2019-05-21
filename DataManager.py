import numpy as np
from sklearn import preprocessing
import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing.imputation import Imputer
import random

class DataManager():

    def __init__(self):
        pass

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

    def preprocessData(self, data):
        imputer = Imputer(missing_values = np.nan, strategy = 'mean')
        imputer.fit(data)
        imputedData = imputer.transform(data) #nan values will take on mean
        scaledData = preprocessing.scale(imputedData).tolist()

        return scaledData

    def assignCodeword(self, labels, codebook):
        labelDictionary = {}
        updatedLabels = []
        random.shuffle(codebook)  # In order to randomize assignment of codewords.

        for originalLabel in labels:
            labelDictionary[originalLabel] = -1
        for label, codeword in zip(labelDictionary, codebook):
            labelDictionary[label] = codeword

        #Updating every original label to its corresponding codeword
        for label in labels:
            updatedLabels.append(labelDictionary[label])

        return updatedLabels, labelDictionary

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

    # Updates the labels to their binary representation for a specific
    # column in the codebook
    def makeTrainingLabels(self, labelDictionaries, labels):
        allUpdatedLabels = []

        for dictionary in labelDictionaries:
            tempLabelList = []
            for label in labels:
                tempLabelList.append(dictionary[label])
            allUpdatedLabels.append(tempLabelList)

        return allUpdatedLabels

    def toCodeword(self, list):
        codeWordList = []
        tempList = []
        counter = 0

        while counter < len(list[0]):
            for prediction in list:
                tempList.append(prediction[counter])
            codeWordList.append(tempList)
            tempList = []
            counter += 1

        return codeWordList

    def originalY(self, updatedLabels, labelDictionary):
        originalLabels = []
        for label in updatedLabels:
            for key, value in labelDictionary.items():
                if value == label:
                    originalLabels.append(key)

        return originalLabels




import numpy as np
from sklearn import preprocessing
import pandas as pd
from numpy import genfromtxt
import random
from sklearn.preprocessing.imputation import Imputer


class DatasetHandler():

    def __init__(self, codebook):
        self.codebook = codebook

    def getData(self, dataset, labelsColumn, dataBeginIndex, dataEndIndex):
        '''
        Gets data and labels from a given dataset.

        :param dataset: Link or folder path to the dataset you want to use.
        :param labelsColumn: The column (i.e. in the CSV file) containing the labels.
        :param dataBeginIndex: Where the data begins (if data in first column -> index = 0).
        :param dataEndIndex: Where the data ends.
        :return: List of the data of the dataset and list of the corresponding labels.
        '''
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
        '''
        Handle missing values and scale the data (scaling necessary for SVM to function well).

        :param data: All of the original data.
        :return: Data that has been processed.
        '''
        imputer = Imputer(missing_values = np.nan, strategy = 'mean')
        imputer.fit(data)
        imputedData = imputer.transform(data) #nan values will take on mean
        scaledData = preprocessing.scale(imputedData).tolist()

        return scaledData

    def assignCodeword(self, labels):
        '''
        Assigns a codeword from the desginated codebook to the labels of the dataset.

        :param labels: The labels of the dataset.
        :return: List of labels that are now codewords and a dictionary with key = original label and value = codeword
        '''
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

    def binarizeLabels(self, labelDictionary):
        '''
        Binarize labels based off of the assigned codeword for each class.
        For training, we need to pass a list of "updated labels" (with the only options being 0 or 1). Whether or not an
        original label is determined to be a 0 or 1 is based off of the columns of the codebook. This method goes down
        each column of the codebook, and creates a list of what the original label should be for a particular column
        of a codebook.

        :param labelDictionary: The dictionary that cotains the original label and its bit representation.
        :return: A list of lists with each nested list containing the binarized labels for a particular column in the
                 codebook.
        '''
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
        '''
        Goes through a dataset and flags all the indices of classes that have a small number of samples.

        :param data: All original data.
        :param labels: All original labels.
        :return: The indices that need to be removed in order to ignore small classes. Additionally, it returns lists
                 containing that data and their corresponding labels (in case you ever want to use it for some reason or
                validate that you're actually retrieving the correct samples).
        '''
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
        '''
        Use the list of indices to remove generated from the 'getSmallClasses()' method to create an updated list of
        data and labels that no longer contain these small classes.

        :param data: All original data.
        :param labels: All original labels.
        :param indicesToRemove: Indices that correspond to the positions of classes that have a small number of samples.
        :return: A list for data and labels that don't contain the small classes.
        '''
        sortedIndicies = sorted(indicesToRemove, reverse=True)
        for index in sortedIndicies:
            del labels[index]
            del data[index]

        return data, labels

    def getHoldoutIndices(self, labels, labelsToRemove):
        '''
        Determine the indices of labels that need to be used as the holdout (used to ensure that we don't use a class that
        has been marked as small and therefore removed).

        :param labels: All original labels
        :param labelsToRemove: Labels that will be removed.
        :return: A list of indices that correspond to the position of classes/labels that will then be used as a holdout.
        '''
        uniqueLabels = np.unique(labels)
        uniqueLabelsToRemove = np.unique(labelsToRemove)
        index = 0
        holdoutIndices = []

        for label in uniqueLabels:
            if label not in uniqueLabelsToRemove:
                holdoutIndices.append(index)
            index += 1

        return holdoutIndices


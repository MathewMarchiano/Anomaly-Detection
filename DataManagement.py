import numpy as np
from sklearn import preprocessing
import pandas as pd
from numpy import genfromtxt
import random
from sklearn.preprocessing.imputation import Imputer
import matplotlib.pyplot as plt

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

# Manages everything to do with finding the optimal threshold. DOES NOT deal with visualizing or interpreting the
# data. DataProcessor will deal with that.
class ThresholdManager():

    def __init__(self):
        pass

    # Get accuracy for known and unknown detection. Known predictions should have a hamming distance
    # value less than that of the threshold value. Unknown predictions should have a value greater than
    # that of the threshold (ideally). Also calculates the absolute value of the difference between the
    # two accuracies which is needed to determine what the most optimal threshold is.
    def getAccuracy(self, threshold, knownHammingDistances, UnknownHammingDistances):
        # HD = 'hamming distance'
        totalKnownHDs = len(knownHammingDistances)
        totalUnknownHDs = len(UnknownHammingDistances)
        knownCounter = 0
        unknownCounter = 0

        for knownHD in knownHammingDistances:
            if knownHD < threshold:
                knownCounter += 1

        for unknownHD in UnknownHammingDistances:
            if unknownHD > threshold:
                unknownCounter += 1

        unknownAccuracy = 1.0 * unknownCounter / totalUnknownHDs
        knownAccuracy = 1.0 * knownCounter / totalKnownHDs
        difference = abs(knownAccuracy - unknownAccuracy)

        return knownAccuracy, unknownAccuracy, difference

    # When finding the optimal threshold, the knownHDs in this context refer to the known validation data
    # (AKA, the 20% of the data withheld from training). The unknownHDs refers to the data that literally
    # belong to the unknown group/split.
    def findOptimalThreshold(self, listOfThresholds, knownHDs, unknownHDs):
        lowestDifference = 1
        optimalThreshold = 0
        highestKnownAcc = 0
        highestUnknownAcc = 0

        for threshold in listOfThresholds:
            knownAcc, unknownAcc, accuracyDifference = self.getAccuracy(threshold, knownHDs, unknownHDs)
            if accuracyDifference < lowestDifference:
                lowestDifference = accuracyDifference
                optimalThreshold = threshold
                highestKnownAcc = knownAcc
                highestUnknownAcc = unknownAcc

        return optimalThreshold, lowestDifference, highestKnownAcc, highestUnknownAcc


# Deals with processing the data retrieved from training. Includes graphing.
class DataProcessor():
    # Graphs the number of number of instances of a particular hamming distance. Distinguishes between
    # hammings distances that are supposed to be known or unknown. Visual representation of how well
    # the threshold is performing. Shows the accuracy of each type of prediction.
    def graphKnownUnknownPredictions(self, knownHDs, unknownHDs, threshold, codebookNum, split, knownAcc, unknownAcc, seed, holdout):
        bins = np.arange(20)

        thresholdText = "Optimal Threshold: " + str(threshold)
        title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(
            seed) + "\n Holdout: " + str(holdout)
        oldAccruacyText = "Known Classes: (" + str(round(knownAcc, 2)) + ")"
        newAccuracyText = "Unknown Classes: (" + str(round(unknownAcc, 2)) + ")"

        plt.hist([knownHDs], bins=bins - .5, alpha=1, label=oldAccruacyText, ec='black')
        plt.hist([unknownHDs], bins=bins - .5, alpha=0.60, label=newAccuracyText, ec='black', color='green')
        plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
        plt.xlabel("Minimum Hamming Distance")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend(loc='upper right')
        # plt.savefig("D:\ECOC\Pictures\HoldoutClassThresholdComparions_LetterRecognition\OldVsNew\\" + str(holdout) + "_" + str(split)
        #             + "_" + str(threshold) + "_" + str(codebookNum) + "DT.jpg")
        plt.show()
        plt.clf()

    # Graphs the holdout class's HDs against the threshold.
    def graphHoldoutHDs(self, threshold, holdoutHammingDistances, accuracy, codebookNum, split, seed, holdout):
        bins = np.arange(20)

        thresholdText = "Optimal Threshold: " + str(threshold)
        title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(
            seed) + "\n Holdout: " + str(holdout)
        accruacyText = "Accuracy: (" + str(round(accuracy, 2)) + ")"

        plt.hist([holdoutHammingDistances], bins=bins - .5, alpha=1, label=accruacyText, ec='black', color='green')
        plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
        plt.xlabel("Minimum Hamming Distance")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend(loc='upper right')
        # plt.savefig("D:\ECOC\Pictures\HoldoutClassThresholdComparions_LetterRecognition\HoldoutTest\\" + str(holdout) + "_" + str(split)
        #             + "_" + str(threshold) + "_" + str(codebookNum) + "_DT.jpg")
        plt.show()
        plt.clf()

    # Graphs the single data samples' hamming distances against the threshold.
    def singleDataHoldoutHistogram(self, threshold, holdoutHammingDistances, accuracy, codebookNum, split, seed, holdout):
        bins = np.arange(20)

        thresholdText = "Optimal Threshold: " + str(threshold)
        title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(
            seed) + "\n Holdout: " + str(holdout)
        accruacyText = "Accuracy: (" + str(round(accuracy, 2)) + ")"

        plt.hist([holdoutHammingDistances], bins=bins - .5, alpha=1, label=accruacyText, ec='black', color='green')
        plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
        plt.xlabel("Minimum Hamming Distance")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend(loc='upper right')
        # plt.savefig("D:\ECOC\Pictures\HoldoutClassThresholdComparions_LetterRecognition\OldHoldoutTest\\" + str(holdout) + "_" + str(split)
        #             + "_" + str(threshold) + "_" + str(codebookNum) + "_DT.jpg")
        plt.show()
        plt.clf()

    # Calculates the accuracy of the predictions made on the holdout class's data. Ideally, the hamming
    # distances should all be greater than the value of the threshold, indicating that the prediction belongs
    # to a new class.
    def holdoutClasssThresholdData(self, holdoutHammingDistances, threshold):
        correctPredictionAmount = 0
        total = len(holdoutHammingDistances)
        for hammingDistance in holdoutHammingDistances:
            if hammingDistance > threshold:
                correctPredictionAmount += 1

        return correctPredictionAmount / total

    # Calculates the accuracy of the predictions made on the single samples of data taken from the known
    # data split (used for training). Ideally, the hamming distances should be less than the value of the
    # threshold, indicating that the prediction belongs to a class that the classifier knows.
    def singleDataSampleThresholdData(self, singleDataHammingDistances, threshold):
        correctPredictionAmount = 0
        total = len(singleDataHammingDistances)
        for hammingDistance in singleDataHammingDistances:
            if hammingDistance < threshold:
                correctPredictionAmount += 1

        return correctPredictionAmount / total

    def accuraciesPlot(self, knownAccMinDict, knownAccMaxDict, unknownAccMinDict, unknownAccMaxDict, codeBook):

        plt.plot(list(knownAccMaxDict.keys()), list(knownAccMaxDict.values()), marker='o', color='blue', label='Known Max')
        plt.plot(list(knownAccMinDict.keys()), list(knownAccMinDict.values()), marker='o', color='red', label='Known Min')
        plt.plot(list(unknownAccMinDict.keys()), list(unknownAccMinDict.values()), marker='o', color='green',
                 label='Unknown Min')
        plt.plot(list(unknownAccMaxDict.keys()), list(unknownAccMaxDict.values()), marker='o', color='black',
                 label='Unknown Max')

        plt.title("Known and Unknown Max and Min Accuracies per Split")
        plt.xlabel("Split")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig("D:\ECOC\KnownUnknownAccuracies_MinMax\Abalone\\" + "_SVM_CWLength(" + str(
            len(codeBook[0])) + ").jpg")
        # plt.show()
        plt.clf()

    def meansPlot(self, codeBook, knownMean, unknownMean):
        plt.plot(list(knownMean.keys()), list(knownMean.values()), marker='o', color='blue', label='Known Mean')
        plt.plot(list(unknownMean.keys()), list(unknownMean.values()), marker='o', color='red', label='Unknown Mean')

        plt.title("Mean Accuracy per Split")
        plt.xlabel("Split")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig("D:\ECOC\KnownUnknownAccuracies_Means\Abalone\\" + "_SVM_CWLength(" + str(
            len(codeBook[0])) + ").jpg")
        plt.clf()
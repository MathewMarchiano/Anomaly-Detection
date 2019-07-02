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

    def testAllThresholds(self, listOfThresholds, knownHDs, unknownHDs):
        knownAccuracies = []
        unknownAccuracies = []
        for threshold in listOfThresholds:
            knownAccuracy, unknownAccuracy, difference = self.getAccuracy(threshold, knownHDs, unknownHDs)
            knownAccuracies.append(knownAccuracy)
            unknownAccuracies.append(unknownAccuracy)

        return knownAccuracies, unknownAccuracies

    def averageThresholdAccuracies(self, listOfAccuracies):
        averagedAccuracyList = []
        # listOfAccuracies contains a list for every holdout. Within every list is a list
        # of accuracies for each potential threshold. Bits gives us the number of thresholds there are.
        numThresholdAccuracies = len(listOfAccuracies[0])
        for thresholdAccuracyIndex in range(numThresholdAccuracies):
            tempAccuracyList = []
            for holdoutAccuracyList in listOfAccuracies:
                tempAccuracyList.append(holdoutAccuracyList[thresholdAccuracyIndex])
            averagedAccuracyList.append(np.mean(tempAccuracyList))

        return averagedAccuracyList


# Deals with processing the data retrieved from training. Includes graphing.
class DataProcessor():
    # Graphs the number of instances of a particular hamming distance. Distinguishes between
    # hamming distances that are supposed to be known or unknown.
    # Used for generating a histogram showing the threshold being TESTED.
    def graphThresholdTestHistogram(self, knownHDs, unknownHDs, threshold,
                                    codebookNum, split, knownAcc,
                                    unknownAcc, seed, holdout, allData,
                                    unknownData, knownTrain, codeBook,
                                    knownSingleDataPoints, saveFolderPath, selectedClassifier):
        bins = np.arange(20)
        ax = plt.subplot(111)
        thresholdText = "Optimal Threshold: " + str(threshold)
        title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(
            seed) + "\n Holdout: " + str(holdout)
        oldAccruacyText = "Known Classes: (" + str(round(knownAcc, 2)) + ")"
        newAccuracyText = "Unknown Classes: (" + str(round(unknownAcc, 2)) + ")"

        ax.hist([knownHDs], bins=bins - .5, alpha=1, label=oldAccruacyText, ec='black')
        ax.hist([unknownHDs], bins=bins - .5, alpha=0.60, label=newAccuracyText, ec='black', color='green')
        ax.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
        ax.set_xlabel("Minimum Hamming Distance")
        ax.set_ylabel("Frequency")
        ax.set_title(title)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

        knownData = len(allData) - len(unknownData)
        percentTraining = round((len(knownTrain) / knownData), 2)
        models = ["SVM", "DT", "LDA", "KNN"]
        saveInfo = saveFolderPath + "\\" + models[selectedClassifier - 1] +"_CB"+ str(codebookNum) + "_CWLength(" + str(len(codeBook[0])) \
                   + ")_Holdout" + str(holdout) + "_Split" + str(split) + "_Threshold" \
                   + str(threshold) +"_UnknownHoldoutClasses1_KnownHoldoutSamples" \
                   + str(len(knownSingleDataPoints)) + "_PercentTrainingData" \
                   + str(percentTraining) + ".png"
        plt.savefig(saveInfo, dpi = 300,
                    bbox_extra_artists = (lgd,), bbox_inches = 'tight')
        # plt.show()
        plt.clf()

    # Used to show histograms of the process of building the threshold. Should be used on
    def graphBuildingThresholdHistogram(self, knownHDs, unknownHDs, threshold,
                                        codebookNum, split, knownAcc,
                                        unknownAcc, seed, holdout, allData,
                                        unknownData, knownTrain, codeBook,
                                        knownSingleDataPoints, saveFolderPath, selectedClassifier):
        bins = np.arange(20)
        ax = plt.subplot(111)
        thresholdText = "Optimal Threshold: " + str(threshold)
        title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(
            seed) + "\n Holdout: " + str(holdout)
        oldAccruacyText = "Known Classes: (" + str(round(knownAcc, 2)) + ")"
        newAccuracyText = "Unknown Classes: (" + str(round(unknownAcc, 2)) + ")"

        ax.hist([knownHDs], bins=bins - .5, alpha=1, label=oldAccruacyText, ec='black')
        ax.hist([unknownHDs], bins=bins - .5, alpha=0.60, label=newAccuracyText, ec='black', color='green')
        ax.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
        ax.set_xlabel("Minimum Hamming Distance")
        ax.set_ylabel("Frequency")
        ax.set_title(title)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        knownData = len(allData) - len(unknownData)
        percentTraining = round((len(knownTrain) / knownData), 2)
        models = ["SVM", "DT", "LDA", "KNN"]
        saveInfo = saveFolderPath + "\\" + models[selectedClassifier - 1] + "_CB" + str(
            codebookNum) + "_CWLength(" + str(len(codeBook[0])) \
                   + ")_Holdout" + str(holdout) + "_Split" + str(split) + "_Threshold" \
                   + str(threshold) + "_UnknownHoldoutClasses1_KnownHoldoutSamples" \
                   + str(len(knownSingleDataPoints)) + "_PercentTrainingData" \
                   + str(percentTraining) + ".png"
        plt.savefig(saveInfo, dpi=300,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()
        plt.clf()

    # Graphs the holdout class's HDs against the threshold.
    def graphHoldoutHDs(self, threshold, holdoutHammingDistances,
                        accuracy, codebookNum, split,
                        seed, holdout, saveFolderPath, selectedClassifier):
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
        models = ["SVM", "DT", "LDA", "KNN"]
        plt.savefig(saveFolderPath + "\\" + str(holdout) + "_" + str(split)
                    + "_" + str(threshold) + "_" + str(codebookNum) + "_" + models[selectedClassifier - 1] + ".png")
        # plt.show()
        plt.clf()

    # Graphs the single data samples' hamming distances against the threshold.
    def singleDataHoldoutHistogram(self, threshold, holdoutHammingDistances,
                                   accuracy, codebookNum, split,
                                   seed, holdout, saveFolderPath, selectedClassifier):
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
        models = ["SVM", "DT", "LDA", "KNN"]
        plt.savefig(saveFolderPath + "\\" + str(holdout) + "_" + str(split)
                    + "_" + str(threshold) + "_" + str(codebookNum) + "_" + models[selectedClassifier - 1] +".png")
        plt.show()
        plt.clf()

    # Calculates the accuracy of the predictions made on the holdout class's data. Ideally, the hamming
    # distances should all be greater than the value of the threshold, indicating that the prediction belongs
    # to a new class.
    def unknownThresholdTest(self, holdoutHammingDistances, threshold):
        correctPredictionAmount = 0
        total = len(holdoutHammingDistances)
        for hammingDistance in holdoutHammingDistances:
            if hammingDistance > threshold:
                correctPredictionAmount += 1

        return (correctPredictionAmount / total)

    # Calculates the accuracy of the predictions made on the single samples of data taken from the known
    # data split (used for training). Ideally, the hamming distances should be less than the value of the
    # threshold, indicating that the prediction belongs to a class that the classifier knows.
    def knownThresholdTest(self, singleDataHammingDistances, threshold):
        correctPredictionAmount = 0
        total = len(singleDataHammingDistances)
        for hammingDistance in singleDataHammingDistances:
            if hammingDistance < threshold:
                correctPredictionAmount += 1

        return (correctPredictionAmount / total)

    # Graphs the accuracies (for known/unknown predictions) for each split.
    def accuraciesPlot(self, knownAccMinDict, knownAccMaxDict,
                       unknownAccMinDict, unknownAccMaxDict, knownMean,
                       unknownMean, codeBook, knownTrain, allData,
                       unknownData, knownSingleDataPoints, saveFolderPath, selectedModel,
                       listOfSplits, codebookNum):
        ax = plt.subplot(111)

        # Representation of min and max values.
        ax.plot(list(knownAccMaxDict.keys()), list(knownAccMaxDict.values()), marker='o', color='blue', label='Known Max')
        ax.plot(list(knownAccMinDict.keys()), list(knownAccMinDict.values()), marker='o', color='red', label='Known Min')
        ax.plot(list(unknownAccMinDict.keys()), list(unknownAccMinDict.values()), marker='o', color='green',
                 label='Unknown Min')
        ax.plot(list(unknownAccMaxDict.keys()), list(unknownAccMaxDict.values()), marker='o', color='black',
                 label='Unknown Max')


        # Representation of the means.
        ax.plot(list(knownMean.keys()), list(knownMean.values()), marker='o', color='magenta',
                 label='Known Mean', linestyle = '--')

        ax.plot(list(unknownMean.keys()), list(unknownMean.values()), marker='o', color='cyan',
                 label='Unknown Mean', linestyle = '--')

        plt.xticks(listOfSplits)

        ax.set_title("Prediction Accuracy per Split")
        ax.set_xlabel("Percent Unknown")
        ax.set_ylabel("Accuracy")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

        knownData = len(allData) - len(unknownData)
        percentTraining = round((len(knownTrain)/knownData), 2)
        models = ["SVM", "DT", "LDA", "KNN"] # Corresponds to the number assigned for each classifier in Trainer.py
        # selectedModel has 1 subtracted from it because the models in the Trainer class are assigned with numbers
        # 1 - 4.
        saveInfo = saveFolderPath + "\\" + models[selectedModel - 1] +"_CB" + str(codebookNum) +"_CWLength(" + str(
            len(codeBook[0])) + ")" + "_UnknownHoldoutClasses1_KnownHoldoutSamples" \
            + str(len(knownSingleDataPoints))+ "_PercentTrainingData"+ str(percentTraining)  + ".png"

        plt.savefig(saveInfo, dpi = 300, bbox_extra_artists = (lgd,), bbox_inches = 'tight')
        plt.show()
        plt.clf()

    # Creates an ROC graph. knownAccs used for true positive rate. unknownAccs used for false positive rate.
    # besteknownAccs is used to graph the point on the ROC that corresponds to the accuracy of the threshold
    # that was determined to be most optimal.
    def getROC(self, unknownAccs, knownAccs, split, codeBook, saveFolderPath, selectedModel, bestKnownAccs,
                                                            bestUnknownAccs, averagedOptimalThreshold, codebookNum):
        fprList = []
        bestFPR = 1 - bestKnownAccs
        # False Positive Rate is = 1 - specificity
        for acc in knownAccs:
            fprList.append(1 - acc)
        print(np.max(fprList))
        ax = plt.subplot(111)
        ax.set_title("ROC")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.plot([0,1],[0,1], 'k--')
        ax.axis([0, 1, 0, 1,])
        ax.plot(fprList, unknownAccs, linewidth=2, label=None)
        roundedAvgOptimalThreshold = round(averagedOptimalThreshold, 1)
        roundedBestAcc = round(bestUnknownAccs, 2)
        ax.scatter(bestFPR, bestUnknownAccs, color='red', label="Avg. Optimal Threshold=" + str(roundedAvgOptimalThreshold) + "\nAccuracy=" + str(roundedBestAcc))
        ax.legend(loc='lower right')
        models = ["SVM", "DT", "LDA", "KNN"]
        saveInfo = saveFolderPath + "\\" + models[selectedModel - 1] + "_CB"+ str(codebookNum) + "_CWLength(" + str(
            len(codeBook[0])) + ")" + "_Split(" + str(split) + ")" + ".png"
        plt.savefig(saveInfo)
        plt.show()
        plt.clf()
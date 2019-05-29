import numpy as np
import random
import math

class Splitter():

    def __init__(self):
        pass

    # Utility method used in splitDataAndLabels to handle splitting of the known data.
    # Known data is used for training. Validation used for threshold.
    def knownDataSplit(self, knownData, knownLabels):
        HOLDOUT_AMOUNT = .20
        knownThresholdData = []
        knownThresholdLabels = []
        indicesToDelete = []

        # Randomly selecting 20% of known data
        # Ensuring that each class is represented in the 20% split
        uniqueLabels = np.unique(knownLabels)
        for uniqueLabel in uniqueLabels:
            index = 0
            potentialIndices = []
            # Gathering all indices of data/label pairings for a particular
            # known class
            for label in knownLabels:
                if uniqueLabel == label:
                    potentialIndices.append(index)
                index += 1
            # Randomly selecting 20% of that data to be used as the holdout
            endIndex = len(potentialIndices) - 1
            numDataSamples = math.ceil(len(potentialIndices)*HOLDOUT_AMOUNT)
            for i in range(numDataSamples):
                randIdx = random.randint(0, endIndex)
                chosenIndex = potentialIndices[randIdx]
                while chosenIndex in indicesToDelete:
                    randIdx = random.randint(0, endIndex)
                    chosenIndex = potentialIndices[randIdx]

                knownThresholdData.append(knownData[chosenIndex])
                knownThresholdLabels.append(knownLabels[chosenIndex])
                indicesToDelete.append(chosenIndex)
        sortedIndices = sorted(indicesToDelete, reverse=True)
        for index in sortedIndices:
            del knownData[index]
            del knownLabels[index]

        # Getting single samples of data that the model will not see
        # during training. Used for testing the threshold.
        singleDataSamples = []
        singleDataSamplesLabels = []
        potentialIndicies = []
        indicesToDelete = []
        index = 0
        uniqueKnownLabels = np.unique(knownThresholdLabels).tolist()
        # Get all data for a particular label. Make list of indices.
        for uniqueLabel in uniqueKnownLabels:
            for label in knownThresholdLabels:
                if uniqueLabel == label:
                    potentialIndicies.append(index)
                index += 1
            # Randomly select an index to choose as a data sample.
            randIdx = random.randint(0, (len(potentialIndicies)) - 1) # Random index to choose potential index
            chosenIdx = potentialIndicies[randIdx]

            singleDataSamples.append(knownThresholdData[chosenIdx])
            singleDataSamplesLabels.append(knownThresholdLabels[chosenIdx])
            indicesToDelete.append(chosenIdx)

            index = 0
            potentialIndicies = []

        sortedIndicies = sorted(indicesToDelete, reverse=True)
        for index in sortedIndicies:
            del knownThresholdData[index]
            del knownThresholdLabels[index]

        return knownThresholdData, knownThresholdLabels, singleDataSamples, \
               singleDataSamplesLabels, knownData, knownLabels

    # Determines whether a label will be known, unknown, or holdout.
    def assignLabel(self, trimmedLabels, allLabels, percentUnknown, holdoutIndex):
        uniqueTrimmedLabels = np.unique(trimmedLabels).tolist()
        uniqueLabels = np.unique(allLabels).tolist()
        numUnknown = int(len(uniqueTrimmedLabels) * percentUnknown) # Casting to int because whole numbers are required.

        #Remove holdout class from selection of labels.
        # We select from the unique labels that haven't been trimmed yet because the indices that have been
        # chosen to be holdouts correspond to the list of all unique labels.
        holdoutClass = uniqueLabels[holdoutIndex]
        # Remove the chosen holdout from the list of trimmed labels so that the holdout cannot be chosen as a known or
        # unknown class.
        uniqueTrimmedLabels.remove(holdoutClass)
        # Randomly select which classes will be unknown.
        # We use the trimmed set of labels now becausae we only want to use classes that have enough
        # samples of data.
        listOfUnknownClasses = random.sample(uniqueTrimmedLabels, numUnknown)
        #Create list of what the known classes will be.
        listOfKnownClasses = []
        for label in uniqueTrimmedLabels:
            if label not in listOfUnknownClasses:
                listOfKnownClasses.append(label)

        return listOfUnknownClasses, listOfKnownClasses, holdoutClass

    # Splits the data and labels into separate lists of holdout, known, or unknown.
    # Will also deal with splitting of the known data through use of knownDataSplit().
    def splitDataAndLabels(self, data, allOrigLabels, unknownClasses, holdoutClass):
        holdoutData = []
        holdoutLabels = []
        unknownData = []
        unknownLabels = []
        indicesToDelete = []
        #Separate holdout data from all data
        for index in range(0, len(data)):
            if allOrigLabels[index] == holdoutClass:
                holdoutData.append(data[index])
                holdoutLabels.append(allOrigLabels[index])
                indicesToDelete.append(index)


        #Separate unknown data from all data. What is left will be the known data and labels
        #in both data and allOriginalLabels variables.
        for index in range(0, len(data)):
            if allOrigLabels[index] in unknownClasses:
                unknownData.append(data[index])
                unknownLabels.append(allOrigLabels[index])
                indicesToDelete.append(index)

        #Delete all indices that were in holdout or unknown.
        #Data and allOrigLabels will then be the known data.
        #Deleting with a reversed list so that shifting won't
        #interfere with the process.
        sortedIndicies = sorted(indicesToDelete, reverse = True)
        for index in sortedIndicies:
            del data[index]
            del allOrigLabels[index]

        #Separate known data
        knownValidationData, knownValidationLabels, singleDataSamples, \
        singleDataSamplesLabels, knownData, knownLabels = self.knownDataSplit(data, allOrigLabels)

        return knownValidationData, knownValidationLabels, singleDataSamples, singleDataSamplesLabels, knownData, \
               knownLabels, unknownData, unknownLabels, holdoutData, holdoutLabels

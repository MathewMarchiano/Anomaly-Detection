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
            if numDataSamples == 1:
                numDataSamples += 1
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
        # during training.
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
    def assignLabel(self, labels, percentUnknown, holdoutIndex):
        uniqueLabels = np.unique(labels).tolist()
        numUnknown = int(len(uniqueLabels) * percentUnknown) # Casting to int because whole numbers are required.

        #Remove holdout class from selection of labels.
        holdoutClass = uniqueLabels[holdoutIndex]
        del uniqueLabels[holdoutIndex]

        #Randomly select which classes will be unknown.
        listOfUnknownClasses = random.sample(uniqueLabels, numUnknown)

        #Create list of what the known classes will be.
        listOfKnownClasses = []
        for label in uniqueLabels:
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
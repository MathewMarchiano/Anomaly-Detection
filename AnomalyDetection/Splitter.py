import numpy as np
import random
import math

class Splitter():

    def __init__(self):
        pass

    def holdoutClassSplit(self, holdoutData, holdoutLabels):
        '''
        Splits the holdout class's data so that we can test creating a codeword for an unknown class
        and then see if we can incorporate that new codeword/class into our model effectively (i.e.
        be able to accurately predict that class in the future).

        :param holdoutData: All the data belonging to the holdout class.
        :param holdoutLabels: All of the labels belonging to the holdout class.
        :return: "First run through" and test lists of the holdout class.
        '''
        testData = []
        testLabels = []
        numDataSamples = len(holdoutData)
        numTestSamples = math.ceil(numDataSamples*.80)
        chosenIndices = random.sample(range(0, numDataSamples), numTestSamples)

        for i in chosenIndices:
            testData.append(holdoutData[i])
            testLabels.append(holdoutLabels[i])

        sortedIndicies = sorted(chosenIndices, reverse=True)
        for index in sortedIndicies:
            del holdoutData[index]
            del holdoutLabels[index]

        return testData, testLabels, holdoutData, holdoutLabels

    def knownDataSplit(self, knownData, knownLabels):
        '''
        Utility method used in splitDataAndLabels to handle splitting of the known data.

        :param knownData: Data marked as known.
        :param knownLabels: Labels marked as known.
        :return: All the different splits of the known data and their corresponding labels.
        '''
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

    def assignLabel(self, trimmedLabels, allLabels, percentUnknown, holdoutIndex, percentKnown):
        '''
        Determines whether a label will be known, unknown, or holdout.

        :param trimmedLabels: List of all the labels belonging to classes that have >3 samples of data.
        :param allLabels: Untrimmed list of labels (contains labels belonging to classes deemed 'small').
        :param percentUnknown: Percentage of the classes to be marked as unknown. NOTE: This is percent of classes NOT
                               percent of number of total samples.
        :param holdoutIndex: Index that will be used to select a holdout class.
        :param percentKnown: Percentage of classes to be used as known. If you want to use all of the remaining
                             classes that aren't marked as unknown, pass -1 as the value.
        :return: Lists of labels (unique) that will be used for splitting the "whole" data/labels into known, unknown,
                 and holdout lists necessary for training and validating.
        '''

        uniqueTrimmedLabels = np.unique(trimmedLabels).tolist()
        uniqueLabels = np.unique(allLabels).tolist()

        # Remove holdout class from selection of labels.
        # We select from the unique labels that haven't been trimmed yet because the indices that have been
        # chosen to be holdouts correspond to the indices of the list of all unique labels.
        holdoutClass = uniqueLabels[holdoutIndex]
        # Remove the chosen holdout from the list of trimmed labels so that the holdout cannot be chosen as a known or
        # unknown class.
        uniqueTrimmedLabels.remove(holdoutClass)

        # We calculate numUnknown after the removal of the holdout class
        numUnknown = math.floor(len(uniqueTrimmedLabels) * percentUnknown) # Flooring to ensure no floats (we need ints)

        # Ensure that at least one class is being used as a holdout
        if numUnknown == 0:
            numUnknown = 1


        # Randomly select which classes will be unknown.
        # We use the trimmed set of labels now becausae we only want to use classes that have enough
        # samples of data.
        listOfUnknownClasses = random.sample(uniqueTrimmedLabels, numUnknown)
        #Create list of what the known classes will be.
        listOfKnownClasses = []
        if percentKnown is -1:
            for label in uniqueTrimmedLabels:
                if label not in listOfUnknownClasses:
                    listOfKnownClasses.append(label)
        else:
            numKnownClasses = math.floor(len(uniqueTrimmedLabels) * percentKnown)
            # Ensure there is one known class at minimum
            if numKnownClasses == 0:
                numKnownClasses = 1
            # Make it so that we randomly select the known classes since we aren't using them all now.
            random.shuffle(uniqueTrimmedLabels)
            classCounter = 0
            i = 0
            totalClasses = len(uniqueTrimmedLabels)
            while classCounter < numKnownClasses and i < totalClasses:
                currentClass = uniqueTrimmedLabels[i]
                if currentClass not in listOfUnknownClasses and currentClass is not holdoutClass:
                    listOfKnownClasses.append(currentClass)
                    classCounter += 1
                i += 1

            if(len(listOfKnownClasses) < numKnownClasses):
                print("WARNING: Number of selected known classes is less than the specified percent amount.")
                print("Check to make sure you aren't going above the maximum percent that can be used for the known split "
                      "given the percent of classes used for the unknown split.")
                print("Current Known Split:", percentKnown,
                      "\nCurrent Unknown Split:", percentUnknown)

        return listOfUnknownClasses, listOfKnownClasses, holdoutClass

    def splitDataAndLabels(self, data, allOrigTrimmedLabels, unknownClasses, holdoutClass, selectedKnownClasses):
        '''
        Splits the data and labels into separate lists of holdout, known, or unknown.
        Will also deal with splitting of the known data through use of knownDataSplit().

        :param data: All original data.
        :param allOrigTrimmedLabels: All the labels of valid classes (those that have >3 samples of data).
        :param unknownClasses: List of classes marked as unknown.
        :param holdoutClass: Classes marked as holdout.
        :return: Lists containing data and labels for all necessary splits.
        '''
        holdoutData = []
        holdoutLabels = []
        unknownData = []
        unknownLabels = []
        indicesToDelete = []
        #Separate holdout data from all data
        for index in range(0, len(data)):
            if allOrigTrimmedLabels[index] == holdoutClass:
                holdoutData.append(data[index])
                holdoutLabels.append(allOrigTrimmedLabels[index])
                indicesToDelete.append(index)


        #Separate unknown data from all data. What is left will be the known data and labels
        #in both data and allOriginalLabels variables.
        for index in range(0, len(data)):
            if allOrigTrimmedLabels[index] in unknownClasses:
                unknownData.append(data[index])
                unknownLabels.append(allOrigTrimmedLabels[index])
                indicesToDelete.append(index)

        #Delete all indices that were in holdout or unknown.
        #Data and allOrigLabels will then be the known data.
        #Deleting with a reversed list so that shifting won't
        #interfere with the process.
        sortedIndicies = sorted(indicesToDelete, reverse = True)
        for index in sortedIndicies:
            del data[index]
            del allOrigTrimmedLabels[index]

        # The remaining values in data and allOrigTrimmedLabels belong to classes that are known
        # However, we may not be using all of the known data for our split (we only want to use the data
        # that has been marked as known)
        indicesToDelete = []
        for i in range(len(data)):
            if allOrigTrimmedLabels[i] not in selectedKnownClasses:
                indicesToDelete.append(i)

        sortedIndicies = sorted(indicesToDelete, reverse=True)
        for index in sortedIndicies:
            del data[index]
            del allOrigTrimmedLabels[index]

        #Separate known data
        knownValidationData, knownValidationLabels, singleDataSamples, \
        singleDataSamplesLabels, knownData, knownLabels = self.knownDataSplit(data, allOrigTrimmedLabels)

        return knownValidationData, knownValidationLabels, singleDataSamples, singleDataSamplesLabels, knownData, \
               knownLabels, unknownData, unknownLabels, holdoutData, holdoutLabels

    def reduceThresholdBuildingSamples_AllClasses(self, smallerSplitDataSamples, largerSplitDataSamples, largerSplitLabels):
        '''
        Used to reduce a certain split's (known or unknown split, specifically) number of samples used to build
        the threshold.
        This particular version of reduceThresholdBuildingSamples is labeled "AllClasses" because it will randomly
          select
        from all samples from all classes of the "largerSplitDataSamples/Labels" variables.

        :param smallerSplitDataSamples: Smaller split's data.
        :param largerSplitDataSamples: Larger split's data
        :param largerSplitLabels:  Larger split's labels.
        :return: Abbreviated list of data and labels of the larger split's data and labels.
        '''

        reducedThresholdBuildingData = [] # The shortened list of data samples which will be used to build threshold.
        reducedThresholdBuildingLabels = [] # Same as above, except labels instead of data.
        numSmallerSplitDataSamples = len(smallerSplitDataSamples)
        endIndex = len(largerSplitDataSamples)

        # Generate random indices to get data from in order to build a "reduced list" of
        # samples to use for building the threshold.
        chosenIndices = random.sample(range(0, endIndex), numSmallerSplitDataSamples)

        # Grab the data and labels corresponding to the  chosen indices:
        for index in chosenIndices:
            reducedThresholdBuildingData.append(largerSplitDataSamples[index])
            reducedThresholdBuildingLabels.append(largerSplitLabels[index])

        return reducedThresholdBuildingData, reducedThresholdBuildingLabels

    def reduceThresholdBuildingSamples_FewestClasses(self, smallerSplitDataSamples, largerSplitDataSamples, largerSplitLabels):
        '''
        Used to reduce a certain split's (known or unknown split, specifically) number of samples used to build
        the threshold (same use as above -- different approach).
        This particular version of reduceThresholdBuildingSamples is labeled "FewestClasses" because it will randomly select
        from one class until there is enough samples to match the smaller split of samples. If that particular class does
        not contain enough samples of data, it will randomly select another class to begin choosing samples from. This
        ensures that the minimum amount of classes are used when building the threshold building samples.

        :param smallerSplitDataSamples: Smaller split's data.
        :param largerSplitDataSamples:  Larger split's data.
        :param largerSplitLabels:  Larger split's labels.
        :return: Abbreviated list of data and labels of the larger split's data and labels.
        '''
        reducedThresholdBuildingData = [] # The shortened list of data samples which will be used to build threshold.
        reducedThresholdBuildingLabels = [] # Same as above, except labels instead of data.
        numSmallerSplitDataSamples = len(smallerSplitDataSamples)

        # Get list of all potential classes to choose from
        uniqueClasses = np.unique(largerSplitLabels)

        # Create a dictionary containing each unique class as a key, and a list of all
        # corresponding data samples to that key as the value
        uniqueClassesDictioary = {}
        for key in uniqueClasses:
            uniqueClassesDictioary[key] = []

        # Populate each list for each key with the indices of their particular data/label
        numLargerSplitLabels = len(largerSplitLabels)
        for key in uniqueClassesDictioary:
            for index in range(numLargerSplitLabels):
                if key == largerSplitLabels[index]:
                    uniqueClassesDictioary[key].append(index)

        # Create a list of indices for both data and labels that will be used to build the
        # threshold. Ensuring that the minimum amount of classes are used.
        chosenIndices = []
        for key in uniqueClassesDictioary:
            # First check if we will need to use the entire class. If the class contains fewer number of samples than
            # what we need total, we will.
            if len(uniqueClassesDictioary[key]) + len(chosenIndices) <= numSmallerSplitDataSamples:
                for value in uniqueClassesDictioary[key]:
                    chosenIndices.append(value)
            # If we don't need all the samples, randomly select the remaining amount that we need
            else:
                # Get random indices to select from the list containing the indices we are interested in
                remainingSamplesNeeded = numSmallerSplitDataSamples - len(chosenIndices)
                endIndex = len(uniqueClassesDictioary[key])
                randomlyChosenIndices = random.sample(range(0, endIndex), remainingSamplesNeeded)
                # Add the indices we're interested to the chosenIndices list
                for index in randomlyChosenIndices:
                    chosenIndices.append(uniqueClassesDictioary[key][index])
                break # If the 'else' part of the loop is reached, then no more samples are needed (no need to keep looping).                                                                # get index we're interested in

        # Get the data and labels that correspond to each index
        for index in chosenIndices:
            reducedThresholdBuildingData.append(largerSplitDataSamples[index])
            reducedThresholdBuildingLabels.append(largerSplitLabels[index])

        return reducedThresholdBuildingData, reducedThresholdBuildingLabels

    def knownTrainingDataSplit(self, trainingData, trainingLabels, percentHoldout):
        '''
        Splits the training data so that the classifiers can be trained and then a portion
        of the same classes used for training can be used for developing the threshold.
        :param trainingData: Data used for training the classifiers
        :param trainingLabels: Labels that correspond to the data used for training the classifiers.
        :param percentHoldout: The percent of the training data to be held out for threshold building.
        :return: 2 pairs of data and labels: One for training one for threshold building
        '''
        indicesToDelete = []
        thresholdBuildingData = []
        thresholdBuildingLabels = []
        uniqueLabels = np.unique(trainingLabels)
        for uniqueLabel in uniqueLabels:
            index = 0
            potentialIndices = []
            # Gathering all indices of data/label pairings for a particular
            # known class
            for label in trainingLabels:
                if uniqueLabel == label:
                    potentialIndices.append(index)
                index += 1
            # Randomly selecting 20% of that data to be used as the holdout
            endIndex = len(potentialIndices) - 1
            numDataSamples = math.ceil(len(potentialIndices) * percentHoldout)
            for i in range(numDataSamples):
                randIdx = random.randint(0, endIndex)
                chosenIndex = potentialIndices[randIdx]
                while chosenIndex in indicesToDelete:
                    randIdx = random.randint(0, endIndex)
                    chosenIndex = potentialIndices[randIdx]

                thresholdBuildingData.append(trainingData[chosenIndex])
                thresholdBuildingLabels.append(trainingLabels[chosenIndex])
                indicesToDelete.append(chosenIndex)
        sortedIndices = sorted(indicesToDelete, reverse=True)
        for index in sortedIndices:
            del trainingData[index]
            del trainingLabels[index]

        return trainingData, trainingLabels, thresholdBuildingLabels, thresholdBuildingLabels




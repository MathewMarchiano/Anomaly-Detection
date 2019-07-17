import numpy as np

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
import numpy as np
import HadamardMatrixGeneration as hmg
import itertools

class ThresholdManager():

    def __init__(self):
        pass

    def getAccuracy(self, threshold, knownHammingDistances, UnknownHammingDistances):
        '''
        Get accuracy for known and unknown detection. Known predictions should have a hamming distance
        value less than that of the threshold value. Unknown predictions should have a value greater than
        that of the threshold (ideally). Also calculates the absolute value of the difference between the
        two accuracies which is needed to determine what the most optimal threshold is.

        :param threshold: Threshold being used determine whether a sample of data's HD is known or unknown.
        :param knownHammingDistances: All HDs generated using known data.
        :param UnknownHammingDistances: All HDs generated using unknown data.
        :return: Accuracy of known predictions, unknown predictions, and the difference between those two accuracies.
        '''

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

    def findOptimalThreshold(self, listOfThresholds, knownHDs, unknownHDs):
        '''
        Finds the threshold that is most effective in distinguishing between known and unknown data.

        :param listOfThresholds: All thresholds that will be tested (only one will be best).
        :param knownHDs: HDs generated using known data.
        :param unknownHDs: HDs generated using unknown data.
        :return: The optimal threshold, lowested difference between known and unknown accuracies, and the highest
                 accuracy for known and unknown predictions that corresponds to the best threshold.
        '''

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

    def findOptimalThreshold_CodebookBased_Averaged(self, codebook):
        listOfDistances = []
        allCombos = list(itertools.combinations(codebook, 2))
        for combo in allCombos:
            hammingDistance = hmg.hammingDistance(combo[0], combo[1])
            listOfDistances.append(hammingDistance)

        return sum(listOfDistances) / len(listOfDistances)

    def findOptimalThreshold_CodebookBased_MaxDistance(self, codebook):
        maxDistance = 0
        allCombos = list(itertools.combinations(codebook, 2))
        for combo in allCombos:
            hammingDistance = hmg.hammingDistance(combo[0], combo[1])
            if hammingDistance > maxDistance:
                maxDistance = hammingDistance

        return maxDistance

    def testAllThresholds(self, listOfThresholds, knownHDs, unknownHDs):
        '''
        Tests all thresholds and records the accuracies of each and every one. This is done so that an ROC curve can
        be generated.

        :param listOfThresholds: All thresholds.
        :param knownHDs: HDs generated using known data.
        :param unknownHDs: HDs generated using unknown data.
        :return: List of known and unknown accuracies that are acquired through testing every threshold.
        '''
        knownAccuracies = []
        unknownAccuracies = []
        for threshold in listOfThresholds:
            knownAccuracy, unknownAccuracy, difference = self.getAccuracy(threshold, knownHDs, unknownHDs)
            knownAccuracies.append(knownAccuracy)
            unknownAccuracies.append(unknownAccuracy)

        return knownAccuracies, unknownAccuracies

    def averageThresholdAccuracies(self, listOfAccuracies):
        '''
        Averages the accuracies for known and unknown predictions. This is used because we are collecting accuracies
        accross many holdout classes (thus, we are averaging each threshold accuracy across each holdout). This is
        necessary in order to produce the ROC.

        :param listOfAccuracies: List of accuracies to be averaged.
        :return:
        '''
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

    def unknownThresholdTest(self, holdoutHammingDistances, threshold):
        '''
        Calculates the accuracy of the predictions made on the holdout class's data. Ideally, the hamming
        distances should all be greater than the value of the threshold, indicating that the prediction belongs
        to a new class.

        :param holdoutHammingDistances: HDs generated using the holdout class.
        :param threshold: The threshold that was determined to be most optimal for distinguishing known and unknown data.
        :return: Accuracy of the threshold at correctly identifying unknown data.
        '''
        correctPredictionAmount = 0
        total = len(holdoutHammingDistances)
        for hammingDistance in holdoutHammingDistances:
            if hammingDistance > threshold:
                correctPredictionAmount += 1

        return (correctPredictionAmount / total)

    def knownThresholdTest(self, singleDataHammingDistances, threshold):
        '''
        Calculates the accuracy of the predictions made on the single samples of data taken from the known
        data split (used for training). Ideally, the hamming distances should be less than the value of the
        threshold, indicating that the prediction belongs to a class that the classifier knows.

        :param singleDataHammingDistances: HDs generated using the single data samples from the known data.
        :param threshold: The treshold that was determined to be most optimal for distinguishing known and unknown data.
        :return: Accuracy of the threshold at correctly identifying known data.
        '''
        correctPredictionAmount = 0
        total = len(singleDataHammingDistances)
        for hammingDistance in singleDataHammingDistances:
            if hammingDistance < threshold:
                correctPredictionAmount += 1

        return (correctPredictionAmount / total)
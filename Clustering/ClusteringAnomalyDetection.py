from sklearn.cluster import KMeans
import numpy as np

class ClusteringAnomalyDetection():

    def runAnomalyDetectionTests(self, knownFittingData, knownTestingData, unknownData, thresholds):

        kmeans = KMeans(n_clusters=1).fit(knownFittingData)
        knownDataDistances = kmeans.transform(knownFittingData)
        unknownDataDistances = kmeans.transform(unknownData)
        # The mean of all the distances generated using data that the model should know will
        # act as the threshold determining whether a data input is known or unknown (based off of
        # its calculated distance to the known cluster's centroid).

        # meanKnownDistance = np.mean(knownDataDistances)
        # knownStandardDev = np.std(knownDataDistances)

        threshold = self.buildThreshold(knownDataDistances, unknownDataDistances, thresholds)

        # Test how well the model is at detecting unknown data
        totalUnknownDistances = len(unknownDataDistances)
        flaggedAnomalies = 0

        for distance in unknownDataDistances:
            if distance > threshold:
                flaggedAnomalies += 1

        knownValidationDistances = kmeans.transform(knownTestingData)
        totalKnownValidationDistances = len(knownTestingData)
        flaggedKnowns = 0

        # Test how well the model is at detecting known data
        for distance in knownValidationDistances:
            if distance <= threshold:
                flaggedKnowns += 1

        unknownPredictionAccuracy = 1.0 * flaggedAnomalies / totalUnknownDistances
        knownPredictionAccuracy = 1.0*flaggedKnowns/totalKnownValidationDistances

        return unknownPredictionAccuracy, knownPredictionAccuracy


    def buildThreshold(self, knownDistances, unknownDistances, thresholds):
        meanKnownDistances = np.mean(knownDistances) # Use the mean of  the knownDistances to be used as the starting point
        lowestDifference = 1
        optimalThreshold = -1

        # Test accuracy by adding thresholds to the mean first:
        for threshold in thresholds:
            tempThreshold = meanKnownDistances + threshold
            knownAcc, unknownAcc, accuracyDifference = self.getAccuracy(tempThreshold, knownDistances, unknownDistances)
            if accuracyDifference < lowestDifference:
                lowestDifference = accuracyDifference
                optimalThreshold = tempThreshold

        # Test accuracy by subtracting thresholds to the mean:
        for threshold in thresholds:
            tempThreshold = meanKnownDistances - threshold
            knownAcc, unknownAcc, accuracyDifference = self.getAccuracy(tempThreshold, knownDistances, unknownDistances)
            if accuracyDifference < lowestDifference:
                lowestDifference = accuracyDifference
                optimalThreshold = tempThreshold

        return optimalThreshold

    def getAccuracy(self, threshold, knownDistances, UnknownDistances):
        totalKnownDistances = len(knownDistances)
        totalUnknownDistances = len(UnknownDistances)
        knownCounter = 0
        unknownCounter = 0

        for knownDistance in knownDistances:
            if knownDistance < threshold:
                knownCounter += 1

        for unknownDistance in UnknownDistances:
            if unknownDistance > threshold:
                unknownCounter += 1

        unknownAccuracy = 1.0 * unknownCounter / totalUnknownDistances
        knownAccuracy = 1.0 * knownCounter / totalKnownDistances
        difference = abs(knownAccuracy - unknownAccuracy)

        return knownAccuracy, unknownAccuracy, difference





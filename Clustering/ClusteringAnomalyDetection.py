from sklearn.cluster import KMeans
import numpy as np

class ClusteringAnomalyDetection():

    def runAnomalyDetectionTests(self, knownFittingData, knownTestingData, unknownData):

        kmeans = KMeans(n_clusters=1).fit(knownFittingData)
        knownDataDistances = kmeans.transform(knownFittingData)
        meanKnownDistance = np.mean(knownDataDistances)
        unknownDataDistances = kmeans.transform(unknownData)

        totalUnknownDistances = len(unknownDataDistances)
        flaggedAnomalies = 0
        for distance in unknownDataDistances:
            if distance > meanKnownDistance:
                flaggedAnomalies += 1

        knownValidationDistances = kmeans.transform(knownTestingData)
        totalKnownValidationDistances = len(knownTestingData)
        flaggedKnowns = 0

        for distance in knownValidationDistances:
            if distance <= meanKnownDistance:
                flaggedKnowns += 1

        unknownPredictionAccuracy = 1.0 * flaggedAnomalies / totalUnknownDistances
        knownPredictionAccuracy = 1.0*flaggedKnowns/totalKnownValidationDistances

        return unknownPredictionAccuracy, knownPredictionAccuracy





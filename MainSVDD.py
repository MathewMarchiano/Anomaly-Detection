from AnomalyDetection.DataManagement import DatasetHandler
from AnomalyDetection.Splitter import  Splitter
import ast
import numpy as np
from sklearn.svm import OneClassSVM

# Returns a list of indices that are able to be a holdout class (e.g. they contain >=3 samples of data and won't be
# removed).
def getHoldoutIndices(dataset, labelsColumn, dataBeginIndex, dataEndIndex):
    dh = DatasetHandler([-1])
    data, labels = dh.getData(dataset, labelsColumn, dataBeginIndex, dataEndIndex)
    indicesToRemove, dataToRemove, labelsToRemove = dh.getSmallClasses(data, labels)
    holdoutIndices = dh.getHoldoutIndices(labels, labelsToRemove)
    return holdoutIndices

# Trims the data (removes classes that have < 3 samples), preprocesses it, and then
# creates the list of dictionaries which will be used to reassign the original labels
# of the dataset to their appropriate binary value for a particular classifier (for training).
# NOTE: This processOriginalData method is different than the one in the AnomalyDetection package.
#       (this one doesn't do anything with codebook stuff/binarizing labels based off of them because
#        it is not needed for these specific tests).
def processOriginalData(dataHandler, data, labels):
    indicesToRemove, dataToRemove, labelsToRemove = dataHandler.getSmallClasses(data, labels)
    trimmedAllData, trimmedAllOriginalLabels = dataHandler.removeSmallClasses(data, labels, indicesToRemove)
    scaledData = dataHandler.preprocessData(trimmedAllData)

    return trimmedAllData, trimmedAllOriginalLabels, scaledData

# Parses a text file containing all of the information necessary to run "runAnomalyDetectionTests" in order to
# retrieve all necessary variables. The only detail it doesn't include is the desired classifier to train with.
def parseDatasetInfoFile(textFile):
    parameterValues = []
    with open(textFile, "rt") as myfile:
        for line in myfile:
            parameterValues.append(line.strip())

    param1 = 'cb1'
    param2 = 'cb2'
    param3 = 'cb3'
    param4 = 'datasetPath'
    param5 = 'thresholds'
    param6 = 'splits'
    param7 = 'filePathAcc'
    param8 = 'filePathHDs'
    param9 = 'labelsColumn'
    param10 = 'dataBeginColumn'
    param11 = 'dataEndColumn'
    param12 = 'filePathROC'
    param13 = 'filePathBuildThresholdHistogram'
    param14 = 'filePathConfusionMatrix'
    listOfUnsetParams = [param1, param2, param3, param4, param5, param6, param7,
                         param8, param9, param10, param11, param12, param13, param14]
    paramValueDictionary = {}
    for unsetParam in listOfUnsetParams:
        paramValueDictionary[unsetParam] = -1

    for param, value in zip(listOfUnsetParams, parameterValues):
        paramValueDictionary[param] = value


    codebook1 = ast.literal_eval(paramValueDictionary[param1])
    codebook2 = ast.literal_eval(paramValueDictionary[param2])
    codebook3 = ast.literal_eval(paramValueDictionary[param3])
    datasetPath = paramValueDictionary[param4]
    thresholds = ast.literal_eval(paramValueDictionary[param5])
    for i in range(0, len(thresholds)):
        thresholds[i] = float(thresholds[i])
    splits = ast.literal_eval(paramValueDictionary[param6])
    for i in range(0, len(splits)):
        splits[i] = float(splits[i])
    filePathAccGraph = paramValueDictionary[param7]
    filePathHDsGraph = paramValueDictionary[param8]
    labelsColumn = int(paramValueDictionary[param9])
    dataBeginColumn = int(paramValueDictionary[param10])
    dataEndColumn = int(paramValueDictionary[param11])
    filePathROC = paramValueDictionary[param12]
    filePathBuildingThresholdHistogram = paramValueDictionary[param13]
    filePathConfusionMatrix = paramValueDictionary[param14]

    return codebook1, codebook2, codebook3, datasetPath, thresholds, splits, filePathAccGraph, filePathHDsGraph, \
           labelsColumn, dataBeginColumn, dataEndColumn, filePathROC, filePathBuildingThresholdHistogram, \
           filePathConfusionMatrix

def SVDDTest(dataset, labelCol, beginDataCol, endDataCol, listOfSplits, thresholds):
    cb = [0]  # Only needed for the creation of DatasetHandler object ("real" codebooks  only needed for
              # for anomaly detection testing using ECOC method.
    SP = Splitter()
    dh = DatasetHandler(cb)
    SVDD = OneClassSVM(gamma='auto')

    holdoutIndices = getHoldoutIndices(dataset, labelCol, beginDataCol, endDataCol)
    allData, allOriginalLabels = dh.getData(dataset, labelCol, beginDataCol, endDataCol)
    savedOriginalLabels = allOriginalLabels.copy()  # All labels required for assignLabel() (not trimmed version)
    initTrimmedAllData, initTrimmedAllOriginalLabels, initScaledData = \
                                processOriginalData(dh, allData, allOriginalLabels)

    for split in listOfSplits:
        knownPredictionAccuracies = []
        unknownPredictionAccuracies = []
        for holdoutIndex in holdoutIndices:
            # Working with copies of the data so that we only need to import the data once per
            # Otherwise the data gets changed slightly per run.
            trimmedAllOriginalLabels = initTrimmedAllOriginalLabels.copy()
            scaledData = initScaledData.copy()

            listOfUnknownClasses, listOfKnownClasses, holdoutClass = \
                SP.assignLabel(trimmedAllOriginalLabels, savedOriginalLabels, split, holdoutIndex)

            knownThresholdBuildingData, knownThresholdBuildingLabels, singleDataSamples, singleDataSamplesLabels, knownData, \
            knownLabels, unknownThresholdBuildingData, unknownThresholdBuildingLabels, holdoutData, holdoutLabels \
                = SP.splitDataAndLabels(scaledData, trimmedAllOriginalLabels, listOfUnknownClasses, holdoutClass)


            # Train SVDD model
            SVDD.fit(knownThresholdBuildingData)

            # Test on known data and unknown data
            knownPredictions = SVDD.predict(singleDataSamples).tolist()
            unknownPredictions = SVDD.predict(holdoutData).tolist()

            # Inliners are predicted as 1's by SVDD. Inliners are equivalent to predictions that are being
            # classified as known.
            knownPredictionAccuracy = knownPredictions.count(1)*1.0/len(knownPredictions)

            # Outliers are predicted as -1's by SVDD. Outliers are equivalent to predictions that are being classified
            # as unknown.
            unknownPredictionAccuracy = unknownPredictions.count(-1) * 1.0 / len(unknownPredictions)


            knownPredictionAccuracies.append((knownPredictionAccuracy))
            unknownPredictionAccuracies.append((unknownPredictionAccuracy))
        print("Split:", split)
        print("\tKnown prediction accuracy across all holdouts:", np.mean(knownPredictionAccuracies))
        print("\tUnknown prediction accuracy across all holdouts:", np.mean(unknownPredictionAccuracies))

textFilePath = "D:\ECOC\ECOC_v2\DatasetParameterFiles\ParameterValueFile_Fashion_Hadamard.txt"
codebook1, codebook2, codebook3, datasetPath, thresholds, splits, filePathAccGraph, filePathHDsGraph, \
           labelsColumn, dataBeginColumn, dataEndColumn, filePathROC, filePathBuildingThresholdHistogram, \
           filePathConfusionMatrix = parseDatasetInfoFile(textFilePath)

SVDDTest(datasetPath, labelsColumn, dataBeginColumn, dataEndColumn, splits, thresholds)
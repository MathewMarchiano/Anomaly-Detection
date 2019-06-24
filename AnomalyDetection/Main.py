from AnomalyDetection.DataManagement import DatasetHandler
from AnomalyDetection.DataManagement import ThresholdManager
from AnomalyDetection.DataManagement import DataProcessor
from AnomalyDetection.Splitter import Splitter
from AnomalyDetection.Trainer import Trainer
import numpy as np
import ast
import warnings

warnings.filterwarnings("ignore")
def runAnomalyDetectionTests(listOfCBs, listOfThresholds, listOfNewSplits, dataset,
                             labelCol, beginDataCol, endDataCol, classifier, folderPathAcc, folderPathHDs):

     holdoutIndices = getHoldoutIndices(dataset, labelCol, beginDataCol, endDataCol)
     iterationCount = 1
     optimalThresholds = []
     listOfDifferences = []
     unknownAccuracies = []
     knownAccuracies = []

     codebookNum = 0
     splitter = Splitter()
     trainer = Trainer()
     tm = ThresholdManager()
     dp = DataProcessor()

     for codebook in listOfCBs:
         # Max
         unknownMaxAccDictionary = {}
         knownMaxAccDictionary = {}
         thresholdMaxDictionary = {}
         # Min
         unknownMinAccDictionary = {}
         knownMinAccDictionay = {}
         thresholdMinDictionary = {}
         # Var
         unknownVarDictionary = {}
         knownVarDictionary = {}
         thresholdVarDictionary = {}
         # Means
         unknownMeanDictionary = {}
         knownMeanDictionary = {}
         thresholdMeanDictionary = {}

         dh = DatasetHandler(codebook)
         codebookNum += 1
         for split in listOfNewSplits:
             for holdout in holdoutIndices:
                 allData, allOriginalLabels = dh.getData(dataset, labelCol, beginDataCol, endDataCol)
                 savedOriginalLabels = allOriginalLabels.copy() # All labels required for assignLabel()
                 trimmedAllData, trimmedAllOriginalLabels, scaledData, codewordColumns = \
                                              processOriginalData(dh, allData, allOriginalLabels, savedOriginalLabels)
                 listOfUnknownClasses, listOfKnownClasses, holdoutClass = \
                     splitter.assignLabel(trimmedAllOriginalLabels, savedOriginalLabels, split, holdout)
                 knownValidationData, knownValidationLabels, singleDataSamples, singleDataSamplesLabels, knownData, \
                 knownLabels, unknownData, unknownLabels, holdoutData, holdoutLabels \
                     = splitter.splitDataAndLabels(scaledData, trimmedAllOriginalLabels, listOfUnknownClasses, holdoutClass)
                 knownECOCLabels = trainer.makeTrainingLabels(codewordColumns, knownLabels)
                 listOfClassifiers = trainer.trainClassifiers(knownData, knownECOCLabels, classifier)

                 # Getting predictions on all relevant data:
                 unknownPreds, holdoutClassPreds, singleDataSamplesPreds, knownValidationPreds = \
                                                            getPredictions(unknownData, holdoutData, singleDataSamples,
                                                            knownValidationData, listOfClassifiers, trainer)

                 # Getting the shortest hamming distance that each prediction corresponds to:
                 unknownECOCPreds, unknownHDs = trainer.hammingDistanceUpdater(codebook, unknownPreds)
                 holdoutClassECOCPreds, holdoutClassHDs = trainer.hammingDistanceUpdater(codebook, holdoutClassPreds)
                 singleDataSamplesECOCPreds, singleDataSamplesHDs = trainer.hammingDistanceUpdater(codebook,
                                                                                                singleDataSamplesPreds)
                 knownValidationECOCPreds, knownValidationHDs = trainer.hammingDistanceUpdater(codebook,
                                                                                               knownValidationPreds)

                 optimalThreshold, lowestDifference, highestKnownAcc, highestUnknownAcc = \
                                                 tm.findOptimalThreshold(listOfThresholds, knownValidationHDs, unknownHDs)

                 # Getting accuracies of predictions (whether known or unknown):
                 knownHoldoutDataThresholdAcc = dp.knownThresholdTest(singleDataSamplesHDs, optimalThreshold)
                 unknownHoldoutDataThresholdAcc = dp.unknownThresholdTest(holdoutClassHDs, optimalThreshold)


                 #print("Codebook:", codebookNum, "split:", split, "iteration:", iterationCount)
                 iterationCount += 1

                 optimalThresholds.append(optimalThreshold)
                 listOfDifferences.append(lowestDifference)
                 unknownAccuracies.append(unknownHoldoutDataThresholdAcc)
                 knownAccuracies.append(knownHoldoutDataThresholdAcc)

                 #Graphing to see how threshold is performing:
                 dp.graphKnownUnknownHDs(singleDataSamplesHDs, holdoutClassHDs, optimalThreshold, codebookNum,
                                         split, knownHoldoutDataThresholdAcc, unknownHoldoutDataThresholdAcc,
                                         12, holdoutClass, trimmedAllData, unknownData, knownData, codebook,
                                         singleDataSamples, folderPathHDs, classifier)


             printResults(unknownAccuracies, knownAccuracies, optimalThresholds, codebookNum, split)

             thresholdMaxDictionary[split] = max(optimalThresholds)
             thresholdMinDictionary[split] = min(optimalThresholds)
             thresholdVarDictionary[split] = np.var((optimalThresholds))
             thresholdMeanDictionary[split] = np.mean(optimalThresholds)

             knownMaxAccDictionary[split] = max(knownAccuracies)
             knownMinAccDictionay[split] = min(knownAccuracies)
             knownVarDictionary[split] = np.var(knownAccuracies)
             knownMeanDictionary[split] = np.mean(knownAccuracies)

             unknownMaxAccDictionary[split] = max(unknownAccuracies)
             unknownMinAccDictionary[split] = min(unknownAccuracies)
             unknownVarDictionary[split] = np.var(unknownAccuracies)
             unknownMeanDictionary[split] = np.mean((unknownAccuracies))
             dp.getROC(unknownAccuracies, knownAccuracies)
             optimalThresholds = []
             unknownAccuracies = []
             knownAccuracies = []
             iterationCount = 1

         dp.accuraciesPlot(knownMinAccDictionay, knownMaxAccDictionary, unknownMinAccDictionary,
                           unknownMaxAccDictionary,knownMeanDictionary, unknownMeanDictionary,
                           codebook, knownData, trimmedAllData, unknownData, singleDataSamples,
                           folderPathAcc, classifier, listOfNewSplits)

# Returns a list of indices that are able to be a holdout class (e.g. they contain >=3 samples of data and won't be
# removed).
def getHoldoutIndices(dataset, labelsColumn, dataBeginIndex, dataEndIndex):
    dh = DatasetHandler([-1])
    data, labels = dh.getData(dataset, labelsColumn, dataBeginIndex, dataEndIndex)
    indicesToRemove, dataToRemove, labelsToRemove = dh.getSmallClasses(data, labels)
    holdoutIndices = dh.getHoldoutIndices(labels, labelsToRemove)
    return holdoutIndices

# Prints information about each of the accuracies and thresholds for a particular run.
# The values printed are the values that will be stored
def printResults(unknownAccuracies, knownAccuracies, optimalThresholds, codebookNum, split):
    print("Codebook:", codebookNum, "split:", split)
    print("Mean of Optimal Thresholds:", np.mean(optimalThresholds))
    print("Max Threshold:", max(optimalThresholds))
    print("Min Threshold:", min(optimalThresholds))
    print("Thresholds Variance:", np.var(optimalThresholds), "\n")

    print("Mean of Known Accuracies:", np.mean(knownAccuracies))
    print("Max Known Accuracy:", max(knownAccuracies))
    print("Min Known Accuracy:", min(knownAccuracies))
    print("Known Accuracies Variance:", np.var(knownAccuracies), "\n")

    print("Mean of Unknown Accuracies:", np.mean(unknownAccuracies))
    print("Max Unknown Accuracy:", max(unknownAccuracies))
    print("Min Unknown Accuracy:", min(unknownAccuracies))
    print("Unknown Accuracies Variance:", np.var(unknownAccuracies), "\n")

# Trims the data (removes classes that have < 3 samples), preprocesses it, and then
# creates the list of dictionaries which will be used to reassign the original labels
# of the dataset to their appropriate binary value for a particular classifier (for training).
def processOriginalData(dataHandler, data, labels, savedLabels):
    indicesToRemove, dataToRemove, labelsToRemove = dataHandler.getSmallClasses(data, labels)
    trimmedAllData, trimmedAllOriginalLabels = dataHandler.removeSmallClasses(data, labels, indicesToRemove)
    scaledData = dataHandler.preprocessData(trimmedAllData)
    ECOCLabels, labelDictionary = dataHandler.assignCodeword(savedLabels)
    codewordColumns = dataHandler.binarizeLabels(labelDictionary)

    return trimmedAllData, trimmedAllOriginalLabels, scaledData, codewordColumns

# Gets the list of codeword predictions for all appropriate splits of data.
def getPredictions(unknownData, holdoutData, singleDataSamples, knownValidationData, listOfClassifiers, trainer):
    unknownPreds = trainer.getPredictions(unknownData, listOfClassifiers)
    holdoutClassPreds = trainer.getPredictions(holdoutData, listOfClassifiers)
    singleDataSamplesPreds = trainer.getPredictions(singleDataSamples, listOfClassifiers)
    knownValidationPreds = trainer.getPredictions(knownValidationData, listOfClassifiers)

    return unknownPreds, holdoutClassPreds, singleDataSamplesPreds, knownValidationPreds

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
    listOfUnsetParams = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11]
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

    return codebook1, codebook2, codebook3, datasetPath, thresholds, splits, filePathAccGraph, filePathHDsGraph, \
           labelsColumn, dataBeginColumn, dataEndColumn

codebook1, codebook2, codebook3, datasetPath, thresholds, splits, filePathAccGraph, filePathHDsGraph, \
                    labelsColumn, dataBeginColumn, dataEndColumn = \
                    parseDatasetInfoFile("D:\ECOC\ECOC_v2\DatasetParameterFiles\ParameterValueFile_LowResolutionSpectrometer_Hadamard.txt")
listOfCBs = [codebook1, codebook2, codebook3]

print("Please enter the path to your parameter value file")
parameterValueFile = input().replace('"', '')
print(parameterValueFile)

codebook1, codebook2, codebook3, datasetPath, thresholds, splits, filePathAccGraph, filePathHDsGraph, \
                    labelsColumn, dataBeginColumn, dataEndColumn = \
                    parseDatasetInfoFile(parameterValueFile)
listOfCBs = [codebook1, codebook2, codebook3]

print("Please select which classifier you would like to use:")
print("\tFor SVM, enter 1.")
print("\tFor DT, enter 2.")
print("\tFor LDA, enter 3.")
print("\tFor KNN, enter 4.")
chosenClassifier = int(input())
classifiers = ["SVM", "DT", "LDA", "KNN"]
print(classifiers[chosenClassifier - 1], "chosen.")
print("Running...")
runAnomalyDetectionTests(listOfCBs, thresholds, splits, datasetPath, labelsColumn,
                         dataBeginColumn, dataEndColumn, chosenClassifier, filePathAccGraph, filePathHDsGraph)

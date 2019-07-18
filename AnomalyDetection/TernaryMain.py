from AnomalyDetection.DataManagement import DatasetHandler
from AnomalyDetection.ThresholdManagement import ThresholdManager
from AnomalyDetection.Graphing import Visuals
from AnomalyDetection.Splitter import Splitter
from AnomalyDetection.Trainer import Trainer
import numpy as np
import ast
from AnomalyDetection.TernaryOperations import TernaryOperator

import warnings
warnings.filterwarnings("ignore")

def runAnomalyDetectionTests_Ternary(listOfCBs, listOfThresholds, listOfNewSplits, dataset,
                             labelCol, beginDataCol, endDataCol, classifier, folderPathAcc,
                             folderPathHDs, ROCPath, buildTresholdHistogramPath, confusionMatrixPath, ternarySymbol,
                             percentTernarySymbols):

     # Determine which classes classes to cycle through (ignoring 'small' classes)
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
     vis = Visuals()
     TO = TernaryOperator()

     for codebook in listOfCBs:
         # Update codebook to have ternary symbols
         codebook = TO.generateTernaryCodebook(codebook, ternarySymbol, percentTernarySymbols)

         # All the dictionaries below are used in creating the graph of all
         # the accuracies across all splits (accuraciesPlot())
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
         allData, allOriginalLabels = dh.getData(dataset, labelCol, beginDataCol, endDataCol)
         savedOriginalLabels = allOriginalLabels.copy()  # All labels required for assignLabel() (not trimmed version)
         initTrimmedAllData, initTrimmedAllOriginalLabels, initScaledData, codewordColumns = \
                            processOriginalData(dh, allData, allOriginalLabels, savedOriginalLabels)

         codebookNum += 1
         for split in listOfNewSplits:
             # Used for ROC
             knownAccuraciesToAverage = []
             unknownAccuraciesToAverage = []
             highestKnownAccuracies = []
             highestUnknownAccuracies = []

             # Lists which will contain the data necessary to create a confusion matrix.
             predictions = []
             actuals = []
             for holdout in holdoutIndices:
                 # Working with copies of the data so that we only need to import the data once per
                 # Otherwise the data gets changed slightly per run.
                 trimmedAllData = initTrimmedAllData.copy()
                 trimmedAllOriginalLabels = initTrimmedAllOriginalLabels.copy()
                 scaledData = initScaledData.copy()

                 listOfUnknownClasses, listOfKnownClasses, holdoutClass = \
                     splitter.assignLabel(trimmedAllOriginalLabels, savedOriginalLabels, split, holdout)

                 knownThresholdBuildingData, knownThresholdBuildingLabels, singleDataSamples, singleDataSamplesLabels, knownData, \
                 knownLabels, unknownThresholdBuildingData, unknownThresholdBuildingLabels, holdoutData, holdoutLabels \
                     = splitter.splitDataAndLabels(scaledData, trimmedAllOriginalLabels, listOfUnknownClasses, holdoutClass)

                 # Ensuring number of unknown threshold building data samples never exceeds known data samples
                 if len(unknownThresholdBuildingData) > len(knownThresholdBuildingData):
                     unknownThresholdBuildingData, unknownThresholdBuildingLabels,  = \
                         splitter.reduceThresholdBuildingSamples_FewestClasses(knownThresholdBuildingData,
                                                        unknownThresholdBuildingData, unknownThresholdBuildingLabels)

                 knownCWLabels = trainer.convertLabelToCodeword(codewordColumns, knownLabels)

                 listOfClassifiers = trainer.trainClassifiers_Ternary(knownData, knownCWLabels, classifier, knownLabels,
                                                                      ternarySymbol)

                 # Getting predictions on all relevant data:
                 unknownThresholdBuildingPreds, holdoutClassPreds, singleDataSamplesPreds, knownThresholdBuildingPreds = \
                                            getPredictions(unknownThresholdBuildingData, holdoutData, singleDataSamples,
                                                                 knownThresholdBuildingData, listOfClassifiers, trainer)

                 # Getting the shortest hamming distance that each prediction corresponds to:
                 unknownThresholdBuildingHDs, holdoutClassHDs, singleDataSamplesHDs, knownThresholdBuildingHDs = \
                     getMinimumHammingDistanceLists(trainer, codebook, unknownThresholdBuildingPreds, holdoutClassPreds,
                                                singleDataSamplesPreds, knownThresholdBuildingPreds)

                 optimalThreshold, lowestDifference, highestKnownAcc, highestUnknownAcc = \
                        tm.findOptimalThreshold(listOfThresholds, knownThresholdBuildingHDs, unknownThresholdBuildingHDs)

                 # Updating the predicted codewords. Used for creating confusion matrix (not needed otherwise, yet).
                 unknownECOCPreds, holdoutClassECOCPreds, singleDataSamplesECOCPreds, knownThresholdBuildingECOCPreds = \
                     updatePredictions(trainer, codebook, unknownThresholdBuildingPreds, holdoutClassPreds,
                                   singleDataSamplesPreds, knownThresholdBuildingPreds, optimalThreshold)

                 predictions.append(singleDataSamplesECOCPreds)

                 # Labels aren't converted to codewords yet
                 codewordSDSLabels =\
                     trainer.toCodeword(trainer.convertLabelToCodeword(codewordColumns, singleDataSamplesLabels))
                 actuals.append(codewordSDSLabels)

                 # Graphs histogram showing the process of building the threshold (different "view" of what this method
                 # is showing slightly below).
                 # The final argument "True" is used to determine where this function should save to file (read
                 # function's comment in the DataManagement class to read more).
                 vis.graphThresholdTestHistogram(knownThresholdBuildingHDs, unknownThresholdBuildingHDs, optimalThreshold,
                                                 codebookNum, split, highestKnownAcc,
                                                 highestUnknownAcc, 12, holdout, allData,
                                                 unknownThresholdBuildingData, knownData, codebook,
                                                 singleDataSamples, buildTresholdHistogramPath, classifier, True)

                 # Data for generating ROC
                 knownAccuraciesAll, unknownAccuraciesAll = tm.testAllThresholds(listOfThresholds,
                                                                knownThresholdBuildingHDs, unknownThresholdBuildingHDs)
                 knownAccuraciesToAverage.append(knownAccuraciesAll)
                 unknownAccuraciesToAverage.append(unknownAccuraciesAll)

                 # Getting accuracies of predictions (whether known or unknown):
                 knownHoldoutDataThresholdAcc = tm.knownThresholdTest(singleDataSamplesHDs, optimalThreshold)
                 unknownHoldoutDataThresholdAcc = tm.unknownThresholdTest(holdoutClassHDs, optimalThreshold)

                 iterationCount += 1

                 optimalThresholds.append(optimalThreshold)
                 highestKnownAccuracies.append(highestKnownAcc)
                 highestUnknownAccuracies.append(highestUnknownAcc)
                 listOfDifferences.append(lowestDifference)
                 unknownAccuracies.append(unknownHoldoutDataThresholdAcc)
                 knownAccuracies.append(knownHoldoutDataThresholdAcc)

                 #Graphing to see how threshold is performing/testing threshold visualization:
                 # The final argument "False" is used to determine where this function should save to file (read
                 # function's comment in the DataManagement class to read more).
                 vis.graphThresholdTestHistogram(singleDataSamplesHDs, holdoutClassHDs, optimalThreshold, codebookNum,
                                                split, knownHoldoutDataThresholdAcc, unknownHoldoutDataThresholdAcc,
                                                12, holdoutClass, trimmedAllData, unknownThresholdBuildingData, knownData,
                                                codebook, singleDataSamples, folderPathHDs, classifier, False)


             # ROC
             averagedKnownAccuracies = tm.averageThresholdAccuracies(knownAccuraciesToAverage)
             averagedUnknownAccuracies = tm.averageThresholdAccuracies(unknownAccuraciesToAverage)
             averagedBestKnownAcc = np.mean(highestKnownAccuracies)
             averagedBestUnknownAcc = np.mean(highestUnknownAccuracies)
             averagedBestThreshold = np.mean(optimalThresholds)
             vis.graphROC(averagedUnknownAccuracies, averagedKnownAccuracies, split, codebook, ROCPath,
                          classifier, averagedBestKnownAcc, averagedBestUnknownAcc, averagedBestThreshold, codebookNum)

             # Confusion matrix
             vis.generateConfusionMatrix(predictions, actuals, codebook, confusionMatrixPath, classifier, codebookNum,
                                         split)

             printResults(unknownAccuracies, knownAccuracies, optimalThresholds, codebookNum, split)

             thresholdMaxDictionary[split] = max(optimalThresholds)
             thresholdMinDictionary[split] = min(optimalThresholds)
             thresholdVarDictionary[split] = np.var((optimalThresholds))
             thresholdMeanDictionary[split] = np.mean(optimalThresholds)

             # Used for creating accuracies graph at the end ('accuraciesPlot()')
             knownMaxAccDictionary[split] = max(knownAccuracies)
             knownMinAccDictionay[split] = min(knownAccuracies)
             knownVarDictionary[split] = np.var(knownAccuracies)
             knownMeanDictionary[split] = np.mean(knownAccuracies)

             unknownMaxAccDictionary[split] = max(unknownAccuracies)
             unknownMinAccDictionary[split] = min(unknownAccuracies)
             unknownVarDictionary[split] = np.var(unknownAccuracies)
             unknownMeanDictionary[split] = np.mean((unknownAccuracies))

             optimalThresholds = []
             unknownAccuracies = []
             knownAccuracies = []
             iterationCount = 1

         vis.accuraciesPlot(knownMinAccDictionay, knownMaxAccDictionary, unknownMinAccDictionary,
                           unknownMaxAccDictionary,knownMeanDictionary, unknownMeanDictionary,
                           codebook, knownData, trimmedAllData, unknownThresholdBuildingData, singleDataSamples,
                           folderPathAcc, classifier, listOfNewSplits, codebookNum)

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

# Handles the updating of predicted codewords (i.e. "autocorrecting" a predicted codeword to the codeword in the
# codebook with the shortest Hamming distance). If the codeword has a minimum HD greater than the value of the
# threshold it will be labeled as "unknown" -- which is encoded by the value -1.
def updatePredictions(trainer, codebook, unknownThresholdBuildingPreds, holdoutClassPreds, singleDataSamplesPreds,
                      knownThresholdBuildingPreds, threshold):
    unknownECOCPreds = trainer.hammingDistanceUpdater(codebook, unknownThresholdBuildingPreds, threshold)
    holdoutClassECOCPreds = trainer.hammingDistanceUpdater(codebook, holdoutClassPreds, threshold)
    singleDataSamplesECOCPreds = trainer.hammingDistanceUpdater(codebook, singleDataSamplesPreds, threshold)
    knownValidationECOCPreds = trainer.hammingDistanceUpdater(codebook, knownThresholdBuildingPreds, threshold)

    return unknownECOCPreds, holdoutClassECOCPreds, singleDataSamplesECOCPreds, knownValidationECOCPreds

def getMinimumHammingDistanceLists(trainer, codebook, unknownThresholdBuildingPreds, holdoutClassPreds,
                                   singleDataSamplesPreds, knownThresholdBuildingPreds):
    unknownThresholdBuildingHDs = []
    holdoutClassHDs = []
    singleDataSamplesHDs = []
    knownThresholdBuildingHDs = []

    for prediction in unknownThresholdBuildingPreds:
        unknownThresholdBuildingHDs.append(trainer.getMinimumHammingDistance(codebook, prediction))

    for prediction in holdoutClassPreds:
        holdoutClassHDs.append(trainer.getMinimumHammingDistance(codebook, prediction))

    for prediction in singleDataSamplesPreds:
        singleDataSamplesHDs.append(trainer.getMinimumHammingDistance(codebook, prediction))

    for prediction in knownThresholdBuildingPreds:
        knownThresholdBuildingHDs.append(trainer.getMinimumHammingDistance(codebook, prediction))

    return unknownThresholdBuildingHDs, holdoutClassHDs, singleDataSamplesHDs, knownThresholdBuildingHDs

# Parses a text file containing all of the information necessary to run "runAnomalyDetectionTests" in order to
# retrieve all necessary variables. The only detail it doesn't include is the desired classifier to train with.
# ***********THIS WILL NEED TO BE UPDATED EVERY TIME A NEW PARAMETER IS ADDED TO THE PARAMETER VALUE FILES**************
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


print("Please enter the path to your parameter value file")
parameterValueFile = input().replace('"', '')
print(parameterValueFile)
codebook1, codebook2, codebook3, datasetPath, thresholds, splits, filePathAccGraph, filePathHDsGraph, \
                    labelsColumn, dataBeginColumn, dataEndColumn, ROCPath, buildThresholdHistogramPath, confusionMatrixPath = \
                    parseDatasetInfoFile(parameterValueFile)
listOfCBs = [codebook1, codebook2, codebook3]

print("Please select which classifier you would like to use:")
print("\tFor SVM, enter 1.")
print("\tFor DT, enter 2.")
print("\tFor LDA, enter 3.")
print("\tFor KNN, enter 4.")
print("\tFor Logistic Regression, enter 5.")
print("\tFor Neural Network, enter 6.")
print("\tFor Naive Bayes, enter 7.")
print("\tFor Random Forest, enter 8.")
chosenClassifier = int(input())
classifiers = ["SVM", "DT", "LDA", "KNN", "Logistic Regression", "Neural Network", "Naive Bayes", "Random Forest"]
print(classifiers[chosenClassifier - 1], "chosen.")
print("Running...")
runAnomalyDetectionTests_Ternary(listOfCBs, thresholds, splits, datasetPath, labelsColumn,
                         dataBeginColumn, dataEndColumn, chosenClassifier,
                         filePathAccGraph, filePathHDsGraph, ROCPath, buildThresholdHistogramPath, confusionMatrixPath,
                                 -1, .5)


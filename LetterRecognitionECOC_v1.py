from sklearn import svm
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import random
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


def classSelection(dataset, numberNew, holdouts):
    def newOldSplit(dataset, numNew, holdouts):
        data = pd.read_csv(dataset, header=None)
        num_columns = len(data.columns)

        # All of the scaled data
        npArray_X = genfromtxt(dataset, delimiter=',', usecols=range(1, num_columns - 1))
        npArray_X_Scaled = preprocessing.scale(npArray_X)
        X_List = npArray_X_Scaled.tolist()

        # All of the labels
        labels_List = data.ix[:, 0].tolist()


        # List so we can have a list of all labels available to randomly select from
        label_Selection = []

        for label in labels_List:
            if label not in label_Selection:
                label_Selection.append(label)

        # Making a list to hold labels for "new" data and one for "old" data
        # Classes randomly selected.
        new_Labels = []
        old_Labels = []

        if holdouts == -1:
            holdout = ""
        else:
            holdout = label_Selection[holdouts]


        print("Holdout:", holdout)
        #Removing the holdout from the list of possible labels to be separated:
        label_Selection.remove(holdout)


        while len(new_Labels) < numNew:
            randIndex = random.randint(0, len(label_Selection) - 1)
            if label_Selection[randIndex] not in new_Labels:
                new_Labels.append(label_Selection[randIndex])

        # Populating old_Labels with everything else
        for label in label_Selection:
            if label not in new_Labels:
                old_Labels.append(label)


       # Getting split data/labels
        all_New_Labels = []
        all_Old_Labels = []
        old_Data = []
        new_Data = []
        wholeClassHoldoutData = []
        wholeClassHoldoutLabels = []
        for index in range(len(labels_List)):
            if labels_List[index] in new_Labels:
                all_New_Labels.append(labels_List[index])
                new_Data.append(X_List[index])
            elif labels_List[index] in old_Labels:
                all_Old_Labels.append(labels_List[index])
                old_Data.append(X_List[index])
            elif labels_List[index] == holdout:
                wholeClassHoldoutData.append(X_List[index])
                wholeClassHoldoutLabels.append(labels_List[index])

        # Split of the known data
        holdoutAmount = len(old_Data)*.2
        old_Holdout_Data = []
        old_Holdout_Labels = []
        counter = 0


        while (counter < holdoutAmount):
            randomIndex = random.randint(0, (len(old_Data) - 1))
            old_Holdout_Data.append(old_Data[randomIndex])
            old_Holdout_Labels.append(all_Old_Labels[randomIndex])
            del old_Data[randomIndex]
            del all_Old_Labels[randomIndex]

            counter += 1

        singleOldDataSamples = []
        singleDataSamplesLabels = []

        for oldLabel in old_Labels:
            randomIndex = random.randint(0, (len(old_Data) - 1))

            while all_Old_Labels[randomIndex] != oldLabel:
                randomIndex = random.randint(0, (len(old_Data) - 1))

            singleOldDataSamples.append(old_Data[randomIndex])
            singleDataSamplesLabels.append((all_Old_Labels[randomIndex]))
            del old_Data[randomIndex]
            del all_Old_Labels[randomIndex]

        return all_New_Labels, all_Old_Labels, old_Data, new_Data, labels_List, old_Holdout_Data, old_Holdout_Labels, wholeClassHoldoutData, wholeClassHoldoutLabels, holdout, singleOldDataSamples, singleDataSamplesLabels


    newLabels, oldLabels, oldData, newData, allLabels, oldHoldoutData, oldHoldoutLabels, wholeClassHoldoutData, wholeClassHoldoutLabels, holdout, singleDataSamples, singleDataSamplesLabels  = newOldSplit(dataset, numberNew, holdouts)
    return newLabels, oldLabels, oldData, newData, allLabels, oldHoldoutData, oldHoldoutLabels, wholeClassHoldoutData, wholeClassHoldoutLabels, holdout, singleDataSamples, singleDataSamplesLabels


def train(codeBook, oldData, newData, allLabels, oldLabels, newLabels, seed, oldHoldoutData, oldHoldoutLabels, wholeClassHoldoutData, wholeClassHoldoutLabels, singleDataSamples, singleDataLabels):
    def codeWordToClassifier(codeBook):
        counter = len(codeBook[0])
        tracker = 0
        classifier = []
        listOfClassifiers = []

        while (tracker < counter):
            for item in codeBook:
                classifier.append(item[tracker])
            listOfClassifiers.append(classifier)
            classifier = []

            tracker += 1

        return listOfClassifiers  # Classifiers that will be used for training

    def Relabel(classifierList, allLabels, newLabels, oldLabels, oldHoldoutLabels, wholeClassHoldoutLabels, singleDataLabels):
        # Temporary Lists
        regular_Temp_List = []
        new_Temp_List = []
        old_Temp_List = []
        holdoutTempList = []
        wholeClassHoldoutTempList = []
        singleDataLabelsTempList = []

        # Master Lists
        regular_Master_List = []
        new_Master_List = []
        old_Master_List = []
        holdoutMasterList = []
        wholeClassHoldoutMasterList = []
        singleDataLabelsMasterList = []

        # Misc
        tracker = 0
        label_Dict = {}
        classifier = 0
        index = 0

        #All labels, regardless of new or old
        for key in allLabels:
            label_Dict[key] = -1

        while (tracker < len(classifierList)):

            # Reassign key value pairing (default -1 value --> classifier values)
            for key in label_Dict:
                label_Dict[key] = classifierList[classifier][index]
                index += 1

            for label in singleDataLabels:
                singleDataLabelsTempList.append(label_Dict[label])

            for label in wholeClassHoldoutLabels:
                wholeClassHoldoutTempList.append(label_Dict[label])

            for label in oldHoldoutLabels:
                holdoutTempList.append(label_Dict[label])

            # Regular list of all labels
            for label in allLabels:
                regular_Temp_List.append(label_Dict[label])

            # Making for new
            for label in newLabels:
                new_Temp_List.append(label_Dict[label])

            # Making for Old
            for label in oldLabels:
                old_Temp_List.append(label_Dict[label])

            singleDataLabelsMasterList.append(singleDataLabelsTempList)
            holdoutMasterList.append(holdoutTempList)
            wholeClassHoldoutMasterList.append(wholeClassHoldoutTempList)
            # Regular "master" list
            regular_Master_List.append(regular_Temp_List)

            # Making "master" list for new labels
            new_Master_List.append(new_Temp_List)

            # Making "master" list for old labels
            old_Master_List.append(old_Temp_List)

            # Resets
            index = 0
            classifier += 1
            tracker += 1
            regular_Temp_List = []
            new_Temp_List = []
            old_Temp_List = []
            holdoutTempList = []
            wholeClassHoldoutTempList = []
            singleDataLabelsTempList = []


        return regular_Master_List, new_Master_List, old_Master_List, holdoutMasterList, wholeClassHoldoutMasterList, singleDataLabelsMasterList

    def Trainer(trainData, trainLabels, newData, holdoutData, seed, wholeClassHoldoutData, singleDataSamples):
        Y_Train_List = []
        Y_Validation_List = []

        # Create lists for training
        # X_Train and X_Validation are always the same (same data)
        for modelNumber in trainLabels:
            X_train, X_validation, Y_train, Y_validation = train_test_split(trainData, modelNumber,
                                                                                             test_size=0.5,
                                                                                             random_state=seed)  # Keeping at a constant value will always return the same
            # value and order of data
            Y_Train_List.append(Y_train)
            Y_Validation_List.append(Y_validation)

        prediction_List = []
        new_Prediction_List = []
        holdoutPredictionList = []
        wholeClassHoldoutPredictionList = []
        singleDataSamplePredictionList = []
        counter = 0
        for trainingSet in Y_Train_List:
            # model = svm.SVC()
            # model = model.fit(X_train, trainingSet)

            model = DecisionTreeClassifier(random_state=0)
            model = model.fit(X_train, trainingSet)

            # model = LinearDiscriminantAnalysis()
            # model = model.fit(X_train, trainingSet)

            # model = KNeighborsClassifier(n_neighbors=2)
            # model = model.fit(X_train, trainingSet)


            predictions = model.predict(X_validation)
            new_Predictions = model.predict(newData)

            if len(holdoutData) > 0:
                holdoutPrediction = model.predict(holdoutData)
                holdoutPredictionList.append(holdoutPrediction)
            if (len(wholeClassHoldoutData) > 0):
                wholeClassHoldoutPrediction = model.predict(wholeClassHoldoutData)
                wholeClassHoldoutPredictionList.append(wholeClassHoldoutPrediction)
            counter += 1


            singleDataSamplePrediction = model.predict(singleDataSamples)
            singleDataSamplePredictionList.append(singleDataSamplePrediction)


            prediction_List.append(predictions)
            new_Prediction_List.append(new_Predictions)


        return prediction_List, Y_Validation_List, new_Prediction_List, holdoutPredictionList, wholeClassHoldoutPredictionList, singleDataSamplePredictionList

    # Constructs predicted codewords
    def predictionToCodeWord(predictionList):
        codeWordList = []
        tempList = []
        counter = 0

        while counter < len(predictionList[0]):
            for prediction in predictionList:
                tempList.append(prediction[counter])
            codeWordList.append(tempList)
            tempList = []
            counter += 1

        return codeWordList

    # Gets accuracy of predicted codewords when compared to
    # actual codewords
    def compare(predictions, actual):
        total = len(predictions)
        right = 0

        for (x, y) in zip(predictions, actual):
            if x == y:
                right += 1

        percentRight = right * 1.0 / total

        return percentRight

    # Will assign each prediction to the best codeword based off of the shortest
    # hamming distance.
    def hammingDistanceUpdater(codeBook, predictedCodewords):
        minHamWord = []
        # List containing actual CW based off of shortest HD
        UpdatedList = []
        minHamList = []
        for predictedCode in predictedCodewords:
            minHam = len(predictedCode)
            for actualCode in codeBook:
                hammingDistance = 0
                for counter in range(0, len(predictedCode)):
                    if actualCode[counter] != predictedCode[counter]:
                        hammingDistance += 1
                if hammingDistance < minHam:
                    minHam = hammingDistance
                    minHamWord = actualCode

            UpdatedList.append(minHamWord)
            minHamList.append(minHam)

        return UpdatedList, minHamList


    classifiers = codeWordToClassifier(codeBook)
    relabeledAllLabels, relabeledNewLabels, relabeledOldLabels, relabeledHoldoutLabels, wholeClassHoldoutLabels, singleDataLabels = Relabel(classifiers, allLabels, newLabels, oldLabels, oldHoldoutLabels, wholeClassHoldoutLabels, singleDataLabels)

    predictionList, actualList, newPredictions, holdoutPredictions, wholeClassHoldoutPredictions, singleDataSamplePredictions = Trainer(oldData, relabeledOldLabels, newData, oldHoldoutData, seed, wholeClassHoldoutData, singleDataSamples)
    predictedCodewords = predictionToCodeWord(predictionList)
    newPredictedCodeWords = predictionToCodeWord(newPredictions)
    holdoutDataCodewords = predictionToCodeWord(holdoutPredictions)
    wholeClassHoldoutDataCodewords = predictionToCodeWord(wholeClassHoldoutPredictions)
    singleDataSamplePredictionCodewords = predictionToCodeWord(singleDataSamplePredictions)
    singleDataSampleValidationCodewords = predictionToCodeWord(singleDataLabels)
    ECOCSingleDataPredictions, singleDataHammingDistances = hammingDistanceUpdater(codeBook, singleDataSamplePredictionCodewords)

    ECOCUpdatedCWs, oldMinHams = hammingDistanceUpdater(codeBook, predictedCodewords)

    # Getting accuracy of old and new split
    newPredsToECOC, newMinHams = hammingDistanceUpdater(codeBook, newPredictedCodeWords)
    holdoutDataPredictionsToECOC, holdoutDataMinHams = hammingDistanceUpdater(codeBook, holdoutDataCodewords)


    wholeClassDataPredictionsToEcoc, wholeClassHoldoutDataMinHams = hammingDistanceUpdater(codeBook, wholeClassHoldoutDataCodewords)

    return oldMinHams, newMinHams, holdoutDataMinHams, seed, wholeClassHoldoutDataMinHams, singleDataHammingDistances, ECOCSingleDataPredictions, singleDataSampleValidationCodewords


#Takes list of minimum hamming distances for the old and new split.
#Will return the accuracy of each split based off of the threshold given.
#Will also return the absolute difference between the old and new accuracies.
def thresholdData(threshold, oldMinHams, newMinHams):
    def thresholdSplit(threshold, oldMinHams, newMinHams):
        oldHamsTotal = len(oldMinHams)
        newHamsTotal = len(newMinHams)
        oldCounter = 0
        newCounter = 0

        for oldHam in oldMinHams:
            if oldHam < threshold:
                oldCounter += 1

        for newHam in newMinHams:
            if newHam > threshold:
                newCounter += 1

        newAccuracy = 1.0 * newCounter / newHamsTotal
        oldAccuracy = 1.0 * oldCounter / oldHamsTotal

        return oldAccuracy, newAccuracy

    def newOldDifference(oldAcc, newAcc):
        return abs(oldAcc - newAcc)

    oldAcc, newAcc = thresholdSplit(threshold, oldMinHams, newMinHams)
    absoluteDifference = newOldDifference(oldAcc, newAcc)

    return absoluteDifference, oldAcc, newAcc




def hamHistogram(oldMinHams, newMinHams, threshold, codebookNum, split, oldAcc, newAcc, seed, holdout):
    bins = np.arange(20)

    thresholdText = "Optimal Threshold: " + str(threshold)
    title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(seed) + "\n Holdout: " + str(holdout)
    oldAccruacyText = "Known Classes: (" + str(round(oldAcc, 2)) + ")"
    newAccuracyText = "Unknown Classes: (" + str(round(newAcc, 2)) + ")"

    plt.hist([oldMinHams], bins = bins -.5, alpha= 1, label=oldAccruacyText, ec = 'black')
    plt.hist([newMinHams], bins = bins -.5, alpha=0.60, label=newAccuracyText, ec = 'black', color = 'green')
    plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
    plt.xlabel("Minimum Hamming Distance")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(loc='upper right')
    # plt.savefig("D:\ECOC\Pictures\HoldoutClassThresholdComparions_LetterRecognition\OldVsNew\\" + str(holdout) + "_" + str(split)
    #             + "_" + str(threshold) + "_" + str(codebookNum) + "DT.jpg")
    # plt.show()
    plt.clf()


def holdoutHistogram(threshold, holdoutHammingDistances, accuracy, codebookNum, split, seed, holdout):
    bins = np.arange(20)

    thresholdText = "Optimal Threshold: " + str(threshold)
    title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(seed) + "\n Holdout: " + str(holdout)
    accruacyText = "Accuracy: (" + str(round(accuracy, 2)) + ")"

    plt.hist([holdoutHammingDistances], bins=bins - .5, alpha=1, label=accruacyText, ec='black', color = 'green')
    plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
    plt.xlabel("Minimum Hamming Distance")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(loc='upper right')
    # plt.savefig("D:\ECOC\Pictures\HoldoutClassThresholdComparions_LetterRecognition\HoldoutTest\\" + str(holdout) + "_" + str(split)
    #             + "_" + str(threshold) + "_" + str(codebookNum) + "_DT.jpg")
    # plt.show()
    plt.clf()


def singleDataHoldoutHistogram(threshold, holdoutHammingDistances, accuracy, codebookNum, split, seed, holdout):
    bins = np.arange(20)

    thresholdText = "Optimal Threshold: " + str(threshold)
    title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(seed) + "\n Holdout: " + str(holdout)
    accruacyText = "Accuracy: (" + str(round(accuracy, 2)) + ")"

    plt.hist([holdoutHammingDistances], bins=bins - .5, alpha=1, label=accruacyText, ec='black', color = 'green')
    plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
    plt.xlabel("Minimum Hamming Distance")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(loc='upper right')
    # plt.savefig("D:\ECOC\Pictures\HoldoutClassThresholdComparions_LetterRecognition\OldHoldoutTest\\" + str(holdout) + "_" + str(split)
    #             + "_" + str(threshold) + "_" + str(codebookNum) + "_DT.jpg")
    # plt.show()
    plt.clf()

def holdoutClasssThresholdData(holdoutHammingDistances, threshold):
    correctPredictionAmount = 0
    total = len(holdoutHammingDistances)
    for hammingDistance in holdoutHammingDistances:
        if hammingDistance > threshold:
            correctPredictionAmount += 1

    return correctPredictionAmount/total


def singleDataSampleThresholdData(singleDataHammingDistances, threshold):
    correctPredictionAmount = 0
    total = len(singleDataHammingDistances)
    for hammingDistance in singleDataHammingDistances:
        if hammingDistance < threshold:
            correctPredictionAmount += 1

    return correctPredictionAmount / total

#Used for converting predictions and validation codewords to a form that scikit learn
#can use for creating a confusion matrix
def convertListOfLists(listOfLists):
    intList = []
    tempString = ""
    counter = 0

    while counter < len(listOfLists):
        for pred in listOfLists[counter]:
            tempString += str(pred)

        intList.append(tempString)
        tempString = ""
        counter += 1

    return intList



#Work in progress
# def averageConfusionMatrix(confusionMatrices):
#     # averagedConfusionMatrix = np.array(confusionMatrices[0])
#     originalSize = len(confusionMatrices)
#     dimension = len(confusionMatrices[0])
#     averagedConfusionMatrix = np.zeros((dimension, dimension))
#
#     for matrix in confusionMatrices:
#         npMatrix = np.array(matrix)
#         column = np.array(matrix).shape[1]
#         if column != len(averagedConfusionMatrix):
#             print("Column:",column, "dimension:", dimension)
#             print(npMatrix)
#             # np.delete(npMatrix, 1, len(averagedConfusionMatrix) - 1)
#         averagedConfusionMatrix = averagedConfusionMatrix + npMatrix
#
#     averagedConfusionMatrix -= confusionMatrices[0]
#     averagedConfusionMatrix = averagedConfusionMatrix/originalSize
#
#     return averagedConfusionMatrix


def  accuraciesPlot(oldAccMinDict, oldAccMaxDict, newAccMinDict, newAccMaxDict, codeBook):

    plt.plot(list(oldAccMaxDict.keys()), list(oldAccMaxDict.values()), marker = 'o', color = 'blue', label = 'Known Max')
    plt.plot(list(oldAccMinDict.keys()), list(oldAccMinDict.values()), marker='o', color = 'red', label = 'Known Min')
    plt.plot(list(newAccMinDict.keys()), list(newAccMinDict.values()), marker='o', color = 'green', label = 'Unknown Min')
    plt.plot(list(newAccMaxDict.keys()), list(newAccMaxDict.values()), marker='o', color = 'black', label = 'Unknown Max')

    plt.title("Known and Unknown Max and Min Accuracies per Split")
    plt.xlabel("Split")
    plt.ylabel("Accuracy")
    plt.legend()

    # plt.savefig("D:\ECOC\KnownUnknownAccuracies_MinMax\LetterRecognition\\" + "_DT_CWLength(" + str(len(codeBook[0])) + ").jpg")
    # plt.show()
    plt.clf()

def meansPlot(codeBook, knownMean, unknownMean):
    plt.plot(list(knownMean.keys()), list(knownMean.values()), marker='o', color='blue', label='Known Mean')
    plt.plot(list(unknownMean.keys()), list(unknownMean.values()), marker='o', color='red', label='Unknown Mean')


    plt.title("Mean Accuracy per Split")
    plt.xlabel("Split")
    plt.ylabel("Accuracy")
    plt.legend()

    # plt.savefig("D:\ECOC\KnownUnknownAccuracies_Means\LetterRecognition\\" + "_DT_CWLength(" + str(len(codeBook[0])) + ").jpg")
    plt.clf()



def loopRunECOC(listOfCBs, listOfThresholds, listOfNewSplits, dataset, inputSeed, numClasses):
    bestThresholds = []
    listOfBestDifferences = []
    bestNewAccuracies = []
    bestOldAccuracies = []
    # confusionMatricies = []

    #Max
    newMaxAccDictionary = {}
    oldMaxAccDictionary = {}
    thresholdMaxDictionary = {}
    #Min
    newMinAccDictionary = {}
    oldMinAccDictionay = {}
    thresholdMinDictionary = {}
    #Var
    newVarDictionary = {}
    oldVarDictionary = {}
    thresholdVarDictionary = {}
    #Means
    newMeanDictionary = {}
    oldMeanDictionary = {}
    thresholdMeanDictionary = {}

    codeBookTracker = 0
    for codeBook in listOfCBs:
        codeBookTracker += 1
        for split in listOfNewSplits:
            for holdoutIndex in range(numClasses):
                newLabels, oldLabels, oldData, newData, allLabels, oldHoldoutData, oldHoldoutLabels, WCHoldoutData, WCHoldoutLabels, holdout, singleDataSamples, singleDataLabels = classSelection(dataset, split, holdoutIndex)

                print("********************Split:", split, "new********************")
                random.shuffle(codeBook) # For randomly assigning codewords to labels
                print(np.unique(newLabels))
                oldHams, newHams, holdoutHams, seed, wholeClassHoldoutHams, singleDataHammingDistances, knownHoldoutPredictions, knownHoldoutValidation = train(codeBook, oldData, newData, allLabels, oldLabels, newLabels, inputSeed, oldHoldoutData, oldHoldoutLabels, WCHoldoutData, WCHoldoutLabels, singleDataSamples, singleDataLabels)
                print("\n")
                bestAbsDiff = 1
                bestThreshold = 0
                bestOld = 0
                bestNew = 0

                for threshold in listOfThresholds:
                    absDiff, oldAcc, newAcc = thresholdData(threshold, holdoutHams, newHams)
                    if absDiff < bestAbsDiff:
                        bestAbsDiff = absDiff
                        bestThreshold = threshold
                        bestOld = oldAcc
                        bestNew = newAcc

                hamHistogram(holdoutHams, newHams, bestThreshold, codeBookTracker, split, bestOld, bestNew, seed, holdout)
                wholeNewClassAccuracy = holdoutClasssThresholdData(wholeClassHoldoutHams, bestThreshold)
                singleDataSampleAccuracy = singleDataSampleThresholdData(singleDataHammingDistances, bestThreshold)
                holdoutHistogram(bestThreshold, wholeClassHoldoutHams, wholeNewClassAccuracy, codeBookTracker, split, seed, holdout)
                singleDataHoldoutHistogram(bestThreshold, singleDataHammingDistances, singleDataSampleAccuracy, codeBookTracker, split, seed, holdout)
                bestThresholds.append(bestThreshold)
                listOfBestDifferences.append(bestAbsDiff)
                bestNewAccuracies.append(bestNew)
                bestOldAccuracies.append(bestOld)





            print("Mean of Optimal Thresholds:", np.mean(bestThresholds))
            print("Max Threshold:", max(bestThresholds))
            print("Min Threshold:", min(bestThresholds))
            print("Thresholds Variance:", np.var(bestThresholds), "\n")
            thresholdMaxDictionary[split] = max(bestThresholds)
            thresholdMinDictionary[split] = min(bestThresholds)
            thresholdVarDictionary[split] = np.var((bestThresholds))
            thresholdMeanDictionary[split] = np.mean(bestThresholds)


            print("Mean of Known Accuracies:", np.mean(bestOldAccuracies))
            print("Max Known Accuracy:", max(bestOldAccuracies))
            print("Min Known Accuracy:", min(bestOldAccuracies))
            print("Known Accuracies Variance:", np.var(bestOldAccuracies), "\n")
            oldMaxAccDictionary[split] = max(bestOldAccuracies)
            oldMinAccDictionay[split] = min(bestOldAccuracies)
            oldVarDictionary[split] = np.var(bestOldAccuracies)
            oldMeanDictionary[split] = np.mean(bestOldAccuracies)

            print("Mean of New Accuracies:", np.mean(bestNewAccuracies))
            print("Max Unknown Accuracy:", max(bestNewAccuracies))
            print("Min Unknown Accuracy:", min(bestNewAccuracies))
            print("Unknown Accuracies Variance:", np.var(bestNewAccuracies), "\n")
            newMaxAccDictionary[split] = max(bestNewAccuracies)
            newMinAccDictionary[split] = min(bestNewAccuracies)
            newVarDictionary[split] = np.var(bestNewAccuracies)
            newMeanDictionary[split] = np.mean((bestNewAccuracies))

            bestThresholds = []
            bestOldAccuracies = []
            bestNewAccuracies = []

        accuraciesPlot(oldMinAccDictionay, oldMaxAccDictionary, newMinAccDictionary, newMaxAccDictionary, codeBook)
        meansPlot(codeBook, oldMeanDictionary, newMeanDictionary)

        # Max Resets
        newMaxAccDictionary = {}
        oldMaxAccDictionary = {}
        thresholdMaxDictionary = {}
        # Min Resets
        newMinAccDictionary = {}
        oldMinAccDictionay = {}
        thresholdMinDictionary = {}
        # Var Resets
        newVarDictionary = {}
        oldVarDictionary = {}
        thresholdVarDictionary = {}
        # Means Resets
        newMeanDictionary = {}
        oldMeanDictionary = {}
        thresholdMeanDictionary = {}



codeBookOne = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
   1, 0, 0], [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
   0, 1, 1, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1,
   1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
   0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0, 1,
   0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0], [1, 0, 0,
   0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
  1], [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0,
  0, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
  0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,
  0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0, 0,
  1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1,
  0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0], [1,
  0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
   1, 0], [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
   0, 0, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1,
   1, 1, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1,
   0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 0, 1, 0,
   1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 1,
   1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
  1], [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
  0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1,
  0, 0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
  0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 1, 1, 0, 1, 0,
  0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 0,
  1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1,
  0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
   1, 0], [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1,
   1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
   0, 1, 1, 1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0,
   1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]]

codeBookOneDotFive = [[0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
   1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], [1, 0, 1, 1, 0, 0,
   0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0,
  1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1,
   0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,
   1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
  0], [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1,
  0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0,
  0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
   1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
   1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
  0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0,
  1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
   0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
   0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 0,
   0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
  0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 1, 0, 1, 0, 0,
  1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
   0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
   1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,
  0, 0], [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
  1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1], [0, 1, 0,
  1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
   1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 1, 0, 0, 1,
   1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0,
  1, 1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1,
  0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0,
   0, 0], [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
   0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0,
   1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 0,
  0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
   0, 0, 0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
  0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0], [1, 0,
  1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
   1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 0, 1, 0, 1, 1,
   0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
  1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
  1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
   0, 1, 0], [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
   1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1], [1, 1,
   0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
  1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]]

codeBookTwo = [[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
   1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
  0, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
  0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
   0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1, 1, 0,
   1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0,
  1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,
  1], [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
  1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
   0, 0, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
   1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
  0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 0,
  0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0,
   1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
  1], [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1,
  1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
   0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1,
   1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1,
  0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1,
  0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0,
   1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
  1], [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0,
  0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0,
   1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
   1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 1,
  0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
   1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  1], [0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
  1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1,
   1, 0, 1, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
   0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1,
  0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1,
  0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
   0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
  1], [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1,
  0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
   0, 1, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1,
   1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
  1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 1, 0,
  1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1,
   1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
  1], [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
  0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
   1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
   1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
  0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0,
  1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
   1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
  1], [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1,
  1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0,
   1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
   1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
  0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 1,
  1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
   1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1,
  1], [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,
  0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
   0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
   1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
  0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]]

listOfThresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
listOfSplits = [7, 9, 11]
codeBookList =[codeBookOne, codeBookOneDotFive, codeBookTwo]

trainingDataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"

loopRunECOC(codeBookList, listOfThresholds, listOfSplits, trainingDataset, 12, 1)

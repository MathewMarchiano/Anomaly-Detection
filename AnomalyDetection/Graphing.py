import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt

class Visuals():
    # Graphs the number of instances of a particular hamming distance. Distinguishes between
    # hamming distances that are supposed to be known or unknown.
    # The last parameter "isBuilding" is used to differentiate this function when being used to visualize
    # building the threshold (True) or visualize testing the threshold (False). This ensures that calling this
    # function won't overwrite itself when trying to show 2 different histograms.
    def graphThresholdTestHistogram(self, knownHDs, unknownHDs, threshold,
                                    codebookNum, split, knownAcc,
                                    unknownAcc, seed, holdout, allData,
                                    unknownData, knownTrain, codeBook,
                                    knownSingleDataPoints, saveFolderPath, selectedClassifier, isBuilding):
        bins = np.arange(20)
        ax = plt.subplot(111)
        thresholdText = "Optimal Threshold: " + str(threshold)
        title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(
            seed) + "\n Holdout: " + str(holdout)
        oldAccruacyText = "Known Classes: (" + str(round(knownAcc, 2)) + ")"
        newAccuracyText = "Unknown Classes: (" + str(round(unknownAcc, 2)) + ")"

        ax.hist([knownHDs], bins=bins - .5, alpha=1, label=oldAccruacyText, ec='black')
        ax.hist([unknownHDs], bins=bins - .5, alpha=0.60, label=newAccuracyText, ec='black', color='green')
        ax.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
        ax.set_xlabel("Minimum Hamming Distance")
        ax.set_ylabel("Frequency")
        ax.set_title(title)

        # Setting up the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles, labels = ax.get_legend_handles_labels()
        # For the line below, bbox_to_anchor manages where the legend will be located
        lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

        knownData = len(allData) - len(unknownData)
        percentTraining = round((len(knownTrain) / knownData), 2)
        models = ["SVM", "DT", "LDA", "KNN", "LogisticRegression", "NeuralNetwork", "NaiveBayes", "Random Forest"]
        if isBuilding:
            saveInfo = saveFolderPath + "\\" + "_BuildingThreshold_" + models[selectedClassifier - 1] +"_CB"+ str(codebookNum) + "_CWLength(" + str(len(codeBook[0])) \
                       + ")_Holdout" + str(holdout) + "_Split" + str(split) + "_Threshold" \
                       + str(threshold) +"_UnknownHoldoutClasses1_KnownHoldoutSamples" \
                       + str(len(knownSingleDataPoints)) + "_PercentTrainingData" \
                       + str(percentTraining) + ".png"
        else:
            saveInfo = saveFolderPath + "\\" + "_TestingThreshold_" + models[selectedClassifier - 1] + "_CB" + str(
                codebookNum) + "_CWLength(" + str(len(codeBook[0])) \
                       + ")_Holdout" + str(holdout) + "_Split" + str(split) + "_Threshold" \
                       + str(threshold) + "_UnknownHoldoutClasses1_KnownHoldoutSamples" \
                       + str(len(knownSingleDataPoints)) + "_PercentTrainingData" \
                       + str(percentTraining) + ".png"
        plt.savefig(saveInfo, dpi = 300,
                    bbox_extra_artists = (lgd,), bbox_inches = 'tight')
        # plt.show()
        plt.clf()

    # Graphs the holdout class's HDs against the threshold.
    def graphHoldoutHDs(self, threshold, holdoutHammingDistances,
                        accuracy, codebookNum, split,
                        seed, holdout, saveFolderPath, selectedClassifier):
        bins = np.arange(20)

        thresholdText = "Optimal Threshold: " + str(threshold)
        title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(
            seed) + "\n Holdout: " + str(holdout)
        accruacyText = "Accuracy: (" + str(round(accuracy, 2)) + ")"

        plt.hist([holdoutHammingDistances], bins=bins - .5, alpha=1, label=accruacyText, ec='black', color='green')
        plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
        plt.xlabel("Minimum Hamming Distance")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend(loc='upper right')
        models = ["SVM", "DT", "LDA", "KNN", "LogisticRegression", "NeuralNetwork", "NaiveBayes", "Random Forest"]
        plt.savefig(saveFolderPath + "\\" + str(holdout) + "_" + str(split)
                    + "_" + str(threshold) + "_" + str(codebookNum) + "_" + models[selectedClassifier - 1] + ".png")
        # plt.show()
        plt.clf()

    # Graphs the single data samples' hamming distances against the threshold.
    def singleDataHoldoutHistogram(self, threshold, holdoutHammingDistances,
                                   accuracy, codebookNum, split,
                                   seed, holdout, saveFolderPath, selectedClassifier):
        bins = np.arange(20)

        thresholdText = "Optimal Threshold: " + str(threshold)
        title = "Codebook: " + str(codebookNum) + " Split: " + str(split) + " new" + "\n Seed: " + str(
            seed) + "\n Holdout: " + str(holdout)
        accruacyText = "Accuracy: (" + str(round(accuracy, 2)) + ")"

        plt.hist([holdoutHammingDistances], bins=bins - .5, alpha=1, label=accruacyText, ec='black', color='green')
        plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=3, label=thresholdText)
        plt.xlabel("Minimum Hamming Distance")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend(loc='upper right')
        models = ["SVM", "DT", "LDA", "KNN", "LogisticRegression", "NeuralNetwork", "NaiveBayes", "Random Forest"]
        plt.savefig(saveFolderPath + "\\" + str(holdout) + "_" + str(split)
                    + "_" + str(threshold) + "_" + str(codebookNum) + "_" + models[selectedClassifier - 1] +".png")
        plt.show()
        plt.clf()

    # Graphs the accuracies (for known/unknown predictions) for each split.
    def accuraciesPlot(self, knownAccMinDict, knownAccMaxDict,
                       unknownAccMinDict, unknownAccMaxDict, knownMean,
                       unknownMean, codeBook, knownTrain, allData,
                       unknownData, knownSingleDataPoints, saveFolderPath, selectedModel,
                       listOfSplits, codebookNum):
        ax = plt.subplot(111)

        # Representation of min and max values.
        ax.plot(list(knownAccMaxDict.keys()), list(knownAccMaxDict.values()), marker='o', color='blue', label='Known Max')
        ax.plot(list(knownAccMinDict.keys()), list(knownAccMinDict.values()), marker='o', color='red', label='Known Min')
        ax.plot(list(unknownAccMinDict.keys()), list(unknownAccMinDict.values()), marker='o', color='green',
                 label='Unknown Min')
        ax.plot(list(unknownAccMaxDict.keys()), list(unknownAccMaxDict.values()), marker='o', color='black',
                 label='Unknown Max')


        # Representation of the means.
        ax.plot(list(knownMean.keys()), list(knownMean.values()), marker='o', color='magenta',
                 label='Known Mean', linestyle = '--')

        ax.plot(list(unknownMean.keys()), list(unknownMean.values()), marker='o', color='cyan',
                 label='Unknown Mean', linestyle = '--')

        plt.xticks(listOfSplits)

        ax.set_title("Prediction Accuracy per Split")
        ax.set_xlabel("Percent Unknown")
        ax.set_ylabel("Accuracy")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

        knownData = len(allData) - len(unknownData)
        percentTraining = round((len(knownTrain)/knownData), 2)
        models = ["SVM", "DT", "LDA", "KNN", "LogisticRegression", "NeuralNetwork", "NaiveBayes", "Random Forest"] # Corresponds to the number assigned for each classifier in Trainer.py
        # selectedModel has 1 subtracted from it because the models in the Trainer class are assigned with numbers
        # 1 - 4.
        saveInfo = saveFolderPath + "\\" + models[selectedModel - 1] +"_CB" + str(codebookNum) +"_CWLength(" + str(
            len(codeBook[0])) + ")" + "_UnknownHoldoutClasses1_KnownHoldoutSamples" \
            + str(len(knownSingleDataPoints))+ "_PercentTrainingData"+ str(percentTraining)  + ".png"

        plt.savefig(saveInfo, dpi = 300, bbox_extra_artists = (lgd,), bbox_inches = 'tight')
        plt.show()
        plt.clf()

    # Creates an ROC graph. knownAccs used for true positive rate. unknownAccs used for false positive rate.
    # besteknownAccs is used to graph the point on the ROC that corresponds to the accuracy of the threshold
    # that was determined to be most optimal.
    def graphROC(self, unknownAccs, knownAccs, split, codeBook, saveFolderPath, selectedModel, bestKnownAcc,
                 bestUnknownAcc, averagedOptimalThreshold, codebookNum):
        fprList = []
        bestFPR = 1 - bestKnownAcc
        # False Positive Rate is = 1 - specificity
        for acc in knownAccs:
            fprList.append(1 - acc)
        ax = plt.subplot(111)
        ax.set_title("ROC")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.plot([0,1],[0,1], 'k--')
        ax.axis([0, 1, 0, 1,])
        ax.plot(fprList, unknownAccs, linewidth=2, label=None)
        roundedAvgOptimalThreshold = round(averagedOptimalThreshold, 1)
        roundedBestAcc = round(bestUnknownAcc, 2)
        ax.scatter(bestFPR, bestUnknownAcc, color='red', label="Avg. Optimal Threshold=" + str(roundedAvgOptimalThreshold) + "\nAccuracy=" + str(roundedBestAcc))
        ax.legend(loc='lower right')
        models = ["SVM", "DT", "LDA", "KNN", "LogisticRegression", "NeuralNetwork", "NaiveBayes", "Random Forest"]
        saveInfo = saveFolderPath + "\\" + models[selectedModel - 1] + "_CB"+ str(codebookNum) + "_CWLength(" + str(
            len(codeBook[0])) + ")" + "_Split(" + str(split) + ")" + ".png"
        plt.savefig(saveInfo)
        plt.show()
        plt.clf()


    def generateConfusionMatrix(self, predictedValues, actualValues, codebook, saveFolderPath, selectedModel,
                                        codebookNum, codeBook, split):
        # numUniqueLabels has 1 added to it in order to account for the unknown label
        numUniqueLabels = len(codebook) + 1
        confusionMatrix = []
        # Create an (numUniqueLabels)x(numUniqueLabels) confusion matrix
        for row in range(numUniqueLabels):
            confusionMatrix.append([])
        for row in confusionMatrix:
            for column in range(numUniqueLabels):
                row.append(0)

        for predictions, actuals in zip(predictedValues, actualValues):
            for prediction, actual in zip(predictions, actuals):
                # If prediction is labeled as known, find row and column that it should
                # be graphed at
                if prediction in codebook:
                    rowNumber = codebook.index(actual)
                    columnNumber = codebook.index( prediction)
                    confusionMatrix[rowNumber][columnNumber] += 1
                # Otherwise, the prediction is unknown
                else:
                    rowNumber = codebook.index(actual)
                    confusionMatrix[rowNumber][numUniqueLabels - 1] += 1

        # Show confusion matrix before the averaging
        plt.figure(figsize=(20, 15))
        ax = plt.subplot()
        sn.heatmap(confusionMatrix, annot=True, ax=ax)
        ax.set_xlabel('Predictions')
        ax.set_ylabel('True Values')
        ax.set_title('Confusion Matrix')
        models = ["SVM", "DT", "LDA", "KNN", "LogisticRegression", "NeuralNetwork", "NaiveBayes", "Random Forest"]
        saveInfo = saveFolderPath + "\\" + "_Original_" + models[selectedModel - 1] + "_CB" + str(codebookNum) + "_CWLength(" + str(
            len(codeBook[0])) + ")" + "_Split(" + str(split) + ")" + ".png"
        plt.savefig(saveInfo, dpi=100)
        plt.show()
        plt.clf()

        # Average confusion matrix across row-wise:
        cmDimension = len(confusionMatrix)
        for row in range(cmDimension):
            rowSum = 0
            for column in range(cmDimension):
                rowSum += confusionMatrix[row][column]
            if rowSum > 0:
                for column in range(cmDimension):
                    confusionMatrix[row][column] = round(confusionMatrix[row][column]/rowSum, 2)

        # Show confusion matrix after averaging row-wise
        plt.figure(figsize=(20, 15))
        ax = plt.subplot()
        sn.heatmap(confusionMatrix, annot=True, ax=ax)
        ax.set_xlabel('Predictions')
        ax.set_ylabel('True Values')
        ax.set_title('Confusion Matrix')
        models = ["SVM", "DT", "LDA", "KNN", "LogisticRegression", "NeuralNetwork", "NaiveBayes", "Random Forest"]
        saveInfo = saveFolderPath + "\\" + "_Averaged_" + models[selectedModel - 1] + "_CB" + str(
            codebookNum) + "_CWLength(" + str(
            len(codeBook[0])) + ")" + "_Split(" + str(split) + ")" + ".png"
        plt.savefig(saveInfo, dpi=100)
        plt.show()
        plt.clf()
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

class Trainer():

    def __init__(self):
        pass

    # Use label dictionaries of "binarized" classes to convert all original labels to 0's and 1's based off of what they
    # were assigned bya given classifier (column in codebook).
    def convertLabelToCodeword(self, labelDictionaries, labels):
        allUpdatedLabels = []
        for dictionary in labelDictionaries:
            tempLabelList = []
            for label in labels:
                tempLabelList.append(dictionary[label])
            allUpdatedLabels.append(tempLabelList)

        return allUpdatedLabels

    # Return models so that predictions can be done later.
    def trainClassifiers(self, knownData, knownLabels, model):
        trainedModels = []

        for labels in knownLabels:
            if model == 1:
                classifier = svm.SVC(gamma='auto')
            elif model == 2:
                classifier = DecisionTreeClassifier(random_state=0)
            elif model == 3:
                classifier = LinearDiscriminantAnalysis()
            elif model == 4:
                classifier = KNeighborsClassifier(n_neighbors=15)
            elif model == 5:
                classifier = LogisticRegression(random_state=1)
            elif model == 6:
                classifier = MLPClassifier(random_state=1)
            elif model == 7:
                classifier = GaussianNB()
            elif model == 8:
                classifier = RandomForestClassifier(random_state=0)
            else:
                print("Specify Classifier")
            classifier = classifier.fit(knownData, labels)
            trainedModels.append(classifier)

        return trainedModels

    # Converts list containing multiple numpy arrays to list of lists containing codewords.
    def toCodeword(self, list):
        codeWordList = []
        tempList = []
        counter = 0

        while counter < len(list[0]):
            for prediction in list:
                tempList.append(prediction[counter])
            codeWordList.append(tempList)
            tempList = []
            counter += 1

        return codeWordList

    # Used trained classifiers to get predictions. Predictions will construct codewords.
    def getPredictions(self, validationData, trainedClassifiers):
        predictionList = []

        for classifier in trainedClassifiers:
            predictions = classifier.predict(validationData)
            predictionList.append(predictions)

        predictionList = self.toCodeword(predictionList)

        return predictionList

    # Takes codewords (usually predicted codewords) and "updates" them to whatever codeword they are
    # closest to (with respect to hamming distance) in a given codebook. Will also return a list that
    # shows what the minimum hamming distances were when deciding which codeword to updated the predicted
    # codeword with.
    def hammingDistanceUpdater(self, codebook, predictedCodewords, threshold):
        minHamWord = [] # List because codewords are represented as binary lists
        unknownPrediction = []
        # Predictions labeled as unknown will have a codeword with -1 for all of its indices.
        # Using a for loop running for how many bits there are for a particular codeword so that
        # this generalizes to all codeword sizes
        for i in range(len(predictedCodewords[0])):
            unknownPrediction.append(-1)
        # List containing updated predicted CW based off of the shortest HD to all
        # codewords in the codebook
        updatedCodewordList = []
        for predictedCode in predictedCodewords:
            minHam = len(predictedCode)
            for actualCode in codebook:
                # GETTING MINIMUM HAMMING DISTANCE
                hammingDistance = 0
                for counter in range(0, len(predictedCode)):
                    if actualCode[counter] != predictedCode[counter]:
                        hammingDistance += 1
                if hammingDistance < minHam:
                    minHam = hammingDistance
                    minHamWord = actualCode
            if minHam > threshold:
                minHamWord = unknownPrediction

            updatedCodewordList.append(minHamWord)

        return updatedCodewordList

    # Sole purpose is to get the minimum Hamming distance for a predicted codeword
    # Different than hammingDistanceUpdater() because sometimes we aren't going to want the
    # actual codeword (in the codebook) that corresponds to the shortest HD.
    def getMinimumHammingDistance(self, codebook, predictedCodeword):
        minHam = len(predictedCodeword)
        # Checking all actual codewords to find the one that has the shortest HD
        # to the predicted codeword
        for actualCodeword in codebook:
            hammingDistance = 0
            for index in range(len(predictedCodeword)):
                if actualCodeword[index] != predictedCodeword[index]:
                    hammingDistance += 1
            if hammingDistance < minHam:
                minHam = hammingDistance
        return minHam

    # Gets accuracy of predicted codewords when compared to
    # actual (i.e. validation) codewords
    def compare(self, predictions, actual):
        total = len(predictions)
        right = 0

        for (x, y) in zip(predictions, actual):
            if x == y:
                right += 1

        percentRight = right * 1.0 / total

        return percentRight

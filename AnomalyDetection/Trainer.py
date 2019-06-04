from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

class Trainer():

    def __init__(self):
        pass

    # Use label dictionaries of binarized classes to convert all original labels to 0's and 1's based off of what they
    # were assigned bya given classifier (column in codebook). Used for training every classifier later on.
    def makeTrainingLabels(self, labelDictionaries, labels):
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
                classifier = KNeighborsClassifier(n_neighbors=2)
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
    def hammingDistanceUpdater(self, codebook, codewords):
        minHamWord = []
        # List containing actual CW based off of shortest HD
        UpdatedList = []
        minHamList = []
        for predictedCode in codewords:
            minHam = len(predictedCode)
            for actualCode in codebook:
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

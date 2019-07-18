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

    def convertLabelToCodeword(self, labelDictionaries, labels):
        '''
        Converts "original" labels to codewords.

        :param labelDictionaries: Dictionary containing the original label (key) and its corresponding binary label (value)
        :param labels: Original labels that you want to convert to a codeword.
        :return: List of the codeword representation of a list of original labels.
        '''
        allUpdatedLabels = []
        for dictionary in labelDictionaries:
            tempLabelList = []
            for label in labels:
                tempLabelList.append(dictionary[label])
            allUpdatedLabels.append(tempLabelList)

        return allUpdatedLabels

    def trainClassifiers(self, knownData, knownLabels, model):
        '''
        Trains classifiers for each bit in a codeword. Each classifier will be used to construct codewords later.

        :param knownData: Known data that will be used for training.
        :param knownLabels: Known labels (corresponding to the known data) that will be used for training.
        :param model: The numerical representation (if block below) of a particular classifier that will be used for
                      training.
        :return: A list of trained classifiers that can then be used for constructing codewords.
        '''

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
                classifier = MLPClassifier(random_state=1, hidden_layer_sizes=(50, ))
            elif model == 7:
                classifier = GaussianNB()
            elif model == 8:
                classifier = RandomForestClassifier(random_state=0)
            else:
                print("Specify Classifier")
            classifier = classifier.fit(knownData, labels)
            trainedModels.append(classifier)

        return trainedModels

    def toCodeword(self, listOfRawPredictions):
        '''
        This method constructs codewords from a list containing lists of predictions for each bit/index for a predicted
        codeword.

        BACKGROUND: In the 'getPredictions()' method, each classifier's predictions of 1 or 0 for a particular bit in
        a codeword is recorded in a numpy array. Because this list contains all bits belonging to a particular
        bit/index (i.e. first bit, second bit, etc.) for all bits of every predicted codeword, we cannot simply take
        the output from the 'getPredictions()' method -- we construct the predicted codeword using this method

        A more 'visual' explanation:
        listOfRawPredictions: [[all bits for index 0], [all bits for index 1], ... [all bits for index n]]
        where m = number of bits  in codeword and n = number of predictions made by the classifiers
        codeword_1 = [list[0][0], list[1][0], ... list[m - 1][0]].
        codeword_n = [list[0][n - 1], list[1][n - 1], ... list[m - 1][n - 1]].

        :param listOfRawPredictions: List of predictions for each bit in each codeword.
        :return: List of all predicted codewords.
        '''
        codeWordList = []
        tempList = []
        counter = 0

        while counter < len(listOfRawPredictions[0]):
            for prediction in listOfRawPredictions:
                tempList.append(prediction[counter])
            codeWordList.append(tempList)
            tempList = []
            counter += 1

        return codeWordList

    def getPredictions(self, validationData, trainedClassifiers):
        '''
        Generate codeword predictions based off of data given to the list of classifiers.

        :param validationData: Data to generate predictions on.
        :param trainedClassifiers: List of trained classifiers.
        :return: List of predictions (in the form of codewords).
        '''
        predictionList = []

        for classifier in trainedClassifiers:
            predictions = classifier.predict(validationData)
            predictionList.append(predictions)

        predictionList = self.toCodeword(predictionList)

        return predictionList

    def hammingDistanceUpdater(self, codebook, predictedCodewords, threshold):
        '''
        Takes codewords (usually predicted codewords) and "updates" them to whatever codeword they are
        closest to (with respect to hamming distance) in a given codebook. Will also return a list that
        shows what the minimum hamming distances were when deciding which codeword to updated the predicted
        codeword with.

        :param codebook: Codebook being used to relabel original labels of the dataset.
        :param predictedCodewords: List of predictions made by the list of classifiers.
        :param threshold: Threshold that was used to distinguish between known and unknown data (used to "update" data
                          that is unknown to a list of a -1's).
        :return: List of all "updated"/"autocorrected" codewords.
        '''
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

    def getMinimumHammingDistance(self, codebook, predictedCodeword):
        '''
        Sole purpose is to get the minimum Hamming distance for a predicted codeword
        Different than hammingDistanceUpdater() because sometimes we aren't going to want the
        actual codeword (in the codebook) that corresponds to the shortest HD.\

        :param codebook: Codebook being used to relabel original labels of the dataset.
        :param predictedCodeword: Predicted codewords from the list of trained classifiers.
        :return: List of minimum Hamming distances.
        '''
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

    def compare(self, predictions, actual):
        '''
        Gets accuracy of predicted codewords when compared to
        actual (i.e. validation) codewords

        :param predictions: Updated/autocorrected codewords from the list of classifiers.
        :param actual: List of codewords of what that prediction should've been for a particular sample of data.
        :return: Percent of predictions that were correct.
        '''
        total = len(predictions)
        right = 0

        for (x, y) in zip(predictions, actual):
            if x == y:
                right += 1

        percentRight = right * 1.0 / total

        return percentRight

import numpy as np
import random

# Manages all things that involve use of the ternary symbol
class TernaryOperator():

    def __init__(self):
        pass

    # First iteration: takes a codebook and randomly selects location(s) to add the ternary symbol
    # for each class
    def generateTernaryCodebook(self, codebook, ternarySymbol, numTernarySymbols):
        for row in codebook:
            chosenIndices = random.sample(range(0, len(row)), numTernarySymbols)
            for index in chosenIndices:
                row[index] = ternarySymbol

        return codebook

    def removeMarkedClasses(self, data, labels, ternarySymbol):
        '''
        If a class is marked with a ternary symbol, this method will remove the class's respective label and data
        from being used for training.

        NOTE: The labels being passed to this method should be the total amount of RELABELED labels (i.e. if there
        are 500 samples of data, there should be 500 labels that are either -1, 1, or 0.

        :param data: Data being used to train the classifiers.
        :param labels: "Ternerized" labels (-1, 1, or 0) being used for training a particular classifier.
        :return: Updated data and labels without marked classes' data or labels.
        '''
        indicesToDelete = []
        index = 0

        # Find indices of samples that are marked with the ternary symbol
        for label in labels:
            if label == ternarySymbol:
                indicesToDelete.append(index)
            index += 1

        # Delete indices from data and labels. Convert back to list (from np array) afterwards.
        data = np.delete(data, indicesToDelete).tolist()
        labels = np.delete(labels, indicesToDelete).tolist()

        return data, labels




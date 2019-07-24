from AnomalyDetection.Trainer import Trainer

class IncrementalLearningFunctions:

    def __init__(self):
        pass

    def generateCodeword_Averaged(self, listOfCodewords):
        '''
        Given a list of data and labels, this method will generate a codeword that (hopefully) doesn't already
        exist within the given codebook. This is intended to be used with only holdout classes for this reason.

        This iteration of the method will create the codeword by taking a list of predicted codewords (generated
        using the holdoutData) and then taking an average of those codewords.

        :param trainedClassifiers: List of trained classifiers used to generate codewords.
        :param holdoutData: List of all data belonging to the holdout class.
        :param holdoutLabels: List of all labels belonging to the holdout class.
        :return: A codeword that will be associated with the holdout/unknown class
        '''

        # Generate a codeword based off of the average of all the codewords in the above list:
        unaveragedCodeword = []
        # Get unaveraged codeword list setup
        for bit in listOfCodewords[0]:
            unaveragedCodeword.append(0)

        # Start adding the bits of every generated codeword to the correct index of the
        # averaged codeword
        for codeword in listOfCodewords:
            index = 0
            for bit in codeword:
                unaveragedCodeword[index] += bit
                index += 1

        # Get the average:
        averagedCodeword = []
        numCodewords = len(listOfCodewords)
        for bit in unaveragedCodeword:
            averagedCodeword.append(bit*1.0/numCodewords)

        # Round the values in the averaged codeword above in order to create a usable codeword.
        finalCodeword = []
        for bit in averagedCodeword:
            if bit < .5:
                finalCodeword.append(0)
            else:
                finalCodeword.append(1)

        return finalCodeword

    def generateCodeword_Mode(self, listOfCodewords):
        '''
        Given a list of data and labels, this method will generate a codeword that (hopefully) doesn't already
        exist within the given codebook. This is intended to be used with only holdout classes for this reason.

        This iteration of the method will create the codeword by taking a list of predicted codewords (generated
        using the holdoutData) and then selecting the codeword that occurs most frequently.

        NOTE: Cannot use numpy's argmax for this since it recursively searches lists to count elements (therefore,
              using it will result in either 0 or 1 being the mode, not a list of 0's or 1's).

        :param trainedClassifiers: List of trained classifiers used to generate codewords.
        :param holdoutData: List of all data belonging to the holdout class.
        :param holdoutLabels: List of all labels belonging to the holdout class.
        :return: A codeword that will be associated with the holdout/unknown class
        '''
        # Get occurrences of each codeword generated
        codewordDictionary = {} # Key : Word | Value : Occurrences of that word
        for word in listOfCodewords:
            # You have to convert the codeword to a String because lists are not hashable
            word = [str(bit) for bit in word] # Have to convert the bits to strings first to get join() method
            stringWord = ''.join(word)
            if stringWord in list(codewordDictionary.keys()):
                codewordDictionary[stringWord] += 1
            else:
                codewordDictionary[stringWord] = 1

        # Find which codeword occurred most frequently
        highestFrequency = 0
        for word in codewordDictionary:
            if codewordDictionary[word] > highestFrequency:
                highestFrequency = codewordDictionary[word]
                mostFrequentWord = word
        # Since we had to convert the codeword to a string in order to hash it into the dictionary,
        # we have to convert it back into a list.
        finalWord = []
        for bit in mostFrequentWord:
            finalWord.append(int(bit))

        return finalWord

    def testGeneratedCodeword(self, codebook, trainedClassifiers, generatedCodeword, holdoutData, threshold):
        '''
        Used to test how well the trained classifiers are at generating the codeword developed
        for a new class.

        :param codebook: The original codebook that's being used.
        :param trainedClassifiers: List of trained classifiers used to generate codewords.
        :param generatedCodeword: The codeword generated for the holdout class.
        :param holdoutData: List of data belonging to the holdout class.
        :return: Accuracy of correctly generating codewords that match the newly added class's codeword.
        '''
        trainer = Trainer()

        # Use a copy of the original codebook to not interfere with everything else happening (not entirely sure if
        # doing this is entirely necessary, but doing just as a precaution).
        codebookCopy = codebook.copy()

        # Add the generated codeword to the codebook so that predicted codewords using the test set of the holdout
        # data can be "updated" or "autocorrected" properly using the hammingDistanceUpdater() method.
        codebookCopy.append(generatedCodeword)

        # Get the predictions, update them, then see how many are right
        predictedCodewords = trainer.getPredictions(holdoutData, trainedClassifiers)
        updatedPredictedCodewords = trainer.hammingDistanceUpdater(codebookCopy, predictedCodewords, threshold)

        # Since all of the data belongs to one class, we can just count how many times the generated codeword for
        # that specific class occurs.
        numCorrectPredictions = updatedPredictedCodewords.count(generatedCodeword)

        return numCorrectPredictions/len(updatedPredictedCodewords)


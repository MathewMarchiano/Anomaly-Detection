# *****************************************
# AUTHOR: MOHAMMED SAROSH KHAN
# VERSION: 9 APRIL 2019
# *****************************************
from scipy.linalg import hadamard
import matplotlib.pyplot as hadGraph
import itertools
import numpy
#Produces a Hadamard of 0's and 1's such that n=2^k
def MakeBinaryHadamard(n):
    matrix = []
    for i in range(len(hadamard(n))):
        matrix.append([])
        for entry in hadamard(n)[i]:
            if (entry==-1):
                matrix[i].append(0)
            else:
                matrix[i].append(entry)
    return matrix

#Truncates any given matrix by t rows/columns acting on minor
#Note, minor specifies the vector containing the minor
#(Which by definition gets truncated)
def TruncateMinor(matrix, t, minor):
    a = 0
    w_on = matrix
    while (a<t):
        w_on.remove(w_on[minor])
        temp = transpose(w_on)
        temp.remove(temp[minor])
        w_on = transpose(temp)
        a+=1
    return w_on

#Transposes any given matrix
def transpose(matrix):
    transposed = []
    for i in range(len(matrix[0])):
        transposed.append([])
        for vector in matrix:
            transposed[i].append(vector[i])
    return transposed

#Produces any square matrix of specified length from a Hadamard
def ProduceSquare(length):
    found = 0
    base = 1
    power = 0
    while (found==0):
        base*=2
        power+=1
        if(base>length):
            found = 1
    matrix = MakeBinaryHadamard(base)
    t = base-length
    final = TruncateMinor(matrix, t, 0)
    return final

#Truncates t rows linearly
def truncateRows(matrix, t):
    a = 0
    w_on = matrix
    while (a<t):
        w_on.remove(w_on[0])
        a+=1
    return w_on

#For Writing matrices to file
def writetoFile(matrix, name):
    file = open(str(name)+".txt", "w")
    for b in matrix:
        file.write(str(b) + "\n")
    file.write("\n")


#Helper routine for finding appropriate subsets
#Checks for a column/row of all 1's or 0's
def checkWeightsForAllExtrema(codes):
    columns = transpose(codes)
    rowLength = len(codes[0])
    colLength = len(columns[0])
    for code in codes:
        weight = 0
        for bit in code:
            weight+=bit
        if(weight== rowLength or weight == 0):
            return 0
    for code in columns:
        weight = 0
        for bit in code:
            weight+=bit
        if (weight == colLength or weight ==0):
            return 0
    return 1

#This is for Finding all 1's or 0's and removing them
#Pass in a codebook, and the size (as a number of elements in the codebook)
def ReturnLightMatrix(matrix, size):
    testResult = -1
    for i in (itertools.combinations(matrix, size)):
        print(str(i) + " " + str(transpose(i)))
        if (checkWeightsForAllExtrema(i)==1):
            testResult=1
            return i
    return testResult

#Generates a Matrix of your choosing
#Note: codeLength>= numberClasses for all matrices!
def GenerateMatrix(numberClasses, codeLength):
    squareSeedWidth = numberClasses if (numberClasses>codeLength) \
        else codeLength
    squareSeed = ProduceSquare(squareSeedWidth)
    linearRowTruncation = codeLength-numberClasses
    finalMatrix = truncateRows(squareSeed, linearRowTruncation)
    return finalMatrix

#Same as above but writes to file
def GenerateMatrixFILE(numberClasses, codeLength):
    squareSeedWidth = numberClasses if (numberClasses>codeLength) \
        else codeLength
    squareSeed = ProduceSquare(squareSeedWidth)
    linearRowTruncation = codeLength-numberClasses
    finalMatrix = truncateRows(squareSeed, linearRowTruncation)
    writetoFile(finalMatrix, ""+str(numberClasses) + "x"+str(codeLength))
    return finalMatrix


#--- Testing Routines ---

def hammingDistance(a, b):  # Finds the hamming distance between two codewords
    distance = 0
    #print(len(a), len(b))
    for i in range(len(a)):
        distance += (a[i] ^ b[i])
    return distance

def generateHammingDistances(matrix):
    size = len(matrix)
    distances = []
    for i in range (size-1):
        for j in range((i+1), size):
            distances.append(hammingDistance(matrix[i], matrix[j]))
    return distances

def Graph(yvals):
    xvals = []
    for i in range(len(yvals)):
        xvals.append(i)
    hadGraph.plot(xvals, yvals)
    hadGraph.show()

def findMin(values):
    min = values[0]
    for i in values:
        if (i<min):
            min = i;
    return min

def findMax(values):
    max = values[0]
    for i in values:
        if (i>max):
            max = i;
    return max

def testingData(start, bound):
    lower = start
    upper = start
    values = []
    xVals = []
    while (upper<=bound):
        values.append(findMin(generateHammingDistances(GenerateMatrix(lower, upper))))
        xVals.append(upper)
        upper+=1
    hadGraph.plot(xVals, values)
    hadGraph.xlabel("Length of Codewords")
    hadGraph.ylabel("Minimum Hamming Distance")
    hadGraph.title("Truncation v Distance")
    hadGraph.show()


def hammingWeight(code):
    weight = 0
    for b in code:
        weight+=b
    return weight

def checkWeights(codes, HWmin, HWmax):
    for i in range(len(codes[0])):
        for code in codes:
            weight = hammingWeight(code)
            if (weight>= HWmin and weight<=HWmax):
                return 1
    return 0


def findMaxHammingWeight(matrix):
    max = hammingWeight(matrix[0])
    min = hammingWeight(matrix[0])
    data = []
    for code in matrix:
        weight = hammingWeight(code)
        if (weight>max):
            max = weight
        if(weight<min):
            min = weight
    data.append(min)
    data.append(max)
    return max

from BaselineTesting.BaselinePredictions import Predictor
from BaselineTesting.BaselinePredictions import DataManager
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

pred = Predictor()
dm = DataManager()
models = ["SVM", "DT", "LDA", "KNN", "Logistic Regression", "Neural Network", "Naive Bayes", "Random Forest"]
dataset = "D:\ECOC\DownloadedDatasets\Fashion.csv"

for i in range(1,9):
    for runs in range(1):
        X, y = dm.getData(-1, 0, 783, dataset)
        indicesToRemove, dataToRemove, labelsToRemove = dm.getSmallClasses(X, y)
        X, y = dm.removeSmallClasses(X, y, indicesToRemove)
        X = dm.preprocessData(X)
        train_X, val_X, train_Y, val_Y = train_test_split(X, y, test_size=.20)
        model = pred.trainModel(train_X, train_Y, i)
        print("Getting scores")
        scores = cross_val_score(model, X, y, cv=5)
        print(models[i-1] + ":", np.mean(scores))










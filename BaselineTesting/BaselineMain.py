from BaselineTesting.BaselinePredictions import Predictor
from sklearn.model_selection import train_test_split

pred = Predictor("https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data")


for runs in range(10):
    X, y = pred.getData(1, 1, 16)
    X = pred.preprocessData(X)

    train_X, val_X, train_Y, val_Y = train_test_split(X, y, test_size=.20, random_state=12)

    model = pred.trainModel(train_X, train_Y, 2)

    print(model.score(val_X, val_Y))










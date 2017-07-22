import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class CorrelationMatrix():
    def __init__(self, dataRows):
        self.dataRows = dataRows
        self.correlationMatrix = []

    def get(self):
        if len(self.correlationMatrix) == 0:
            self.correlationMatrix = []
            for rowXIndex, rowX in self.dataRows.items():
                resultRow = []
                for rowYIndex, rowY in self.dataRows.items():
                    correlation = np.corrcoef(rowX, rowY)[0, 1]
                    resultRow.append(correlation)
                self.correlationMatrix.append(resultRow)

        return self.correlationMatrix

    def toString(self):
        matrix = self.get()
        matrixString =""

        matrixString += ("....\t")
        for key in self.dataRows:
            matrixString += '{:4.4}'.format(key)
            matrixString += "\t"
        matrixString += "\n"

        for index, row in enumerate(matrix):
            rowTitle = list(self.dataRows.keys())[index]
            rowString = '{:4.4}'.format(rowTitle) + "\t"

            for field in row:
                rowString += '{:4.4}'.format(str(field))
                rowString += "\t"
            matrixString += rowString 
            matrixString += "\n" 

        return matrixString

class AnalysisInformation():
    def __init__(self, y_pred, y_true):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.support = 0
        (self.accuracy, self.precision, self.recall, self.support) = precision_recall_fscore_support(y_true, y_pred, average='binary') 

    def toString(self):
        return "Accuracy: " + str(self.accuracy) + " Precision: " + str(self.precision) + " Recall: " + str(self.recall)


class ConfusionMatrix():
    def __init__(self, predictions, reality, verbose = False):
        self.predictions = 0
        self.positives = 0
        self.negatives = 0
        self.trueNegatives = 0
        self.truePositives = 0
        self.falseNegatives = 0
        self.falsePositives = 0
        self.correctPredictions = 0
        self.verbose = verbose

        for index, prediction in enumerate(predictions) :
            self.predictions += 1

            if self.verbose:
                print((prediction, reality[index]))
            if prediction == 1:
                self.positives += 1
                if reality[index] == 1:
                    if self.verbose:
                        print("match")
                    self.truePositives += 1
                else:
                    self.falsePositives +=1
            else:
                self.negatives +=1
                if reality[index] == 0:
                    self.trueNegatives += 1
                    if self.verbose:
                        print("match")
                else:
                    self.falseNegatives += 1

        self.correctPredictions = self.truePositives + self.trueNegatives

    # true negative rate
    def getPrecision(self):
        if self.negatives == 0:
            return 0
        return self.trueNegatives/self.negatives

    # true positive rate
    def getRecall(self):
        if self.positives == 0:
            return 0
        return self.truePositives/self.positives

    def getAccuracy(self):
        return self.correctPredictions/self.predictions

    def toString(self):
        return "Accuracy: " + str(self.getAccuracy()) + "\tPrecision: " + str(self.getPrecision()) + "\tRecall: " + str(self.getRecall())

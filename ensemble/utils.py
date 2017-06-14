import numpy as np

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

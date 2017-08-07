import numpy as np
from utils import AnalysisInformation
from sklearn import metrics
from sklearn.model_selection import cross_val_score

class Classifier:
    def __init__(self):
        self.tested = False
        self.calibrated = None
        self.testResult = None

    def getWeights(self, labelArray):
        unique, counts = np.unique(labelArray, return_counts=True)
        classCounts = dict(zip(unique, counts))
        weights = [None] * len(labelArray)

        for key, label in enumerate(labelArray):
            weights[key] = len(labelArray)/classCounts[label]

        return np.array(weights)

    def testFeatureMatrix(self, x, y):
        if not self.tested:
            result = self.calibrated.predict(x)

            prob_pos_isotonic = self.calibrated.predict_proba(x)[:, 1]

            custom_scorer = metrics.make_scorer(AnalysisInformation.analysisInformationScorer)
            scores = cross_val_score(self.calibrated, x, y, cv=10, scoring=custom_scorer)

            analysisInformation = AnalysisInformation(initializer=scores)

            self.testResult = (analysisInformation, result, prob_pos_isotonic)
            self.tested= True
        return self.testResult
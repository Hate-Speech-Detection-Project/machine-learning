from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
from sklearn.svm import NuSVC
from utils import AnalysisInformation
import scipy.sparse as sps

from preprocessor import Preprocessor

class SVMClassifier:
    def __init__(self):
        self.trained = False
        self.tested = False
        self.model = None
        self.calibrated = None
        self.testResult = None

    def fitFeatureMatrix(self, x, y):
        if not self.trained:
            if sps.issparse(x):
                x = x.todense()
            clf = NuSVC()
            self.model = clf.fit(x, y)

            self.calibrated = CalibratedClassifierCV(self.model, cv=2, method='isotonic')
            self.calibrated.fit(x, y)
            self.trained = True
            print("done")

    def testFeatureMatrix(self, x, y):
        if not self.tested:
            print(x)
            print(y)

            if sps.issparse(x):
                x = x.todense()
            # Use the random forest to make sentiment label predictions
            result = self.model.predict(x)

            prob_pos_isotonic = self.calibrated.predict_proba(x)[:, 1]

            analysisInformation = AnalysisInformation(Preprocessor.convertBoolStringsToNumbers(result), Preprocessor.convertBoolStringsToNumbers(y))

            self.testResult = (analysisInformation, result, prob_pos_isotonic)
            self.tested= True
        return self.testResult

    def predict(self, featureMatrix):
        prob_pos_isotonic = self.calibrated.predict_proba(featureMatrix)[:, 1]
        return prob_pos_isotonic[0]

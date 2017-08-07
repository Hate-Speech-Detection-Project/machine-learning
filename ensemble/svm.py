from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
from sklearn.svm import NuSVC
from utils import AnalysisInformation
import scipy.sparse as sps
from classifier import Classifier
from preprocessor import Preprocessor

class SVMClassifier(Classifier):
    def __init__(self):
        self.name = 'SVM'
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
            self.model = clf.fit(x, y, sample_weight=self.getWeights(y))

            self.calibrated = CalibratedClassifierCV(self.model, cv=2, method='isotonic')
            self.calibrated.fit(x, y, sample_weight=self.getWeights(y))
            self.trained = True
            print("done")

    def testFeatureMatrix(self, x, y):
        if not self.tested:
            if sps.issparse(x):
                x = x.todense()
            # Use the random forest to make sentiment label predictions
            result = self.model.predict(x)

            prob_pos_isotonic = self.calibrated.predict_proba(x)[:, 1]

            analysisInformation = AnalysisInformation(result, y)

            self.testResult = (analysisInformation, result, prob_pos_isotonic)
            self.tested= True
        return self.testResult

    def predict(self, featureMatrix):
        if sps.issparse(featureMatrix):
            featureMatrix = featureMatrix.todense()

        prob_pos_isotonic = self.calibrated.predict_proba(featureMatrix)[:, 1]
        return prob_pos_isotonic[0]

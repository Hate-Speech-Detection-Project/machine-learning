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
        super().__init__()
        self.name = 'SVM'
        self.trained = False
        self.model = None
        self.testResult = None

    def fitFeatureMatrix(self, x, y):
        if not self.trained:
            if sps.issparse(x):
                x = x.todense()
            clf = NuSVC()
            self.model = clf.fit(x, y, sample_weight=self.getWeights(y))

            self.calibrated = CalibratedClassifierCV(self.model, method='isotonic', cv=10)
            self.calibrated.fit(x, y, sample_weight=self.getWeights(y))
            self.trained = True
            print("done")

    def predict(self, featureMatrix):
        if sps.issparse(featureMatrix):
            featureMatrix = featureMatrix.todense()

        prob_pos_isotonic = self.calibrated.predict_proba(featureMatrix)[:, 1]
        return prob_pos_isotonic[0]

from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from utils import AnalysisInformation
from classifier import Classifier

from preprocessor import Preprocessor

class AdaBoost(Classifier):
    def __init__(self):
        super().__init__()
        self.name = "AdaBoost"
        self.trained = False
        self.model = None

    def fitFeatureMatrix(self, x, y):
        if not self.trained:
            clf = AdaBoostClassifier(n_estimators=100)
            self.model = clf.fit(x, y, sample_weight=self.getWeights(y))

            self.calibrated = CalibratedClassifierCV(self.model, method='isotonic', cv=10)
            self.calibrated.fit(x, y, sample_weight=self.getWeights(y))
            self.trained = True
            print("done")

    def predict(self, featureMatrix):
        prob_pos_isotonic = self.calibrated.predict_proba(featureMatrix)[:, 1]
        return prob_pos_isotonic[0]

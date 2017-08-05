from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from preprocessor import Preprocessor
from utils import AnalysisInformation

class RandomForestBOWClassifier:
    def __init__(self):
        self.trained = False
        self.tested = False
        self.model = None
        self.calibrated = None
        self.testResult = None

    def fitFeatureMatrix(self, x, y):
        if not self.trained:
            self.model = RandomForestClassifier(n_estimators = 100)
            self.model.fit(x, y)

            self.calibrated = CalibratedClassifierCV(self.model, cv=2, method='isotonic')
            self.calibrated.fit(x, y)
            self.trained = True

    def testFeatureMatrix(self, x, y):
        if not self.tested:
            # Use the random forest to make sentiment label predictions
            result = self.model.predict(x)

            prob_pos_isotonic = self.calibrated.predict_proba(x)[:, 1]

            analysisInformation = AnalysisInformation(result, y)
            self.testResult = (analysisInformation, result, prob_pos_isotonic)
            self.tested = True

        return self.testResult

    def predict(self, featureMatrix):
        return self.calibrated.predict_proba(featureMatrix)[:, 1][0]

 #   def predict(self, comment):
 #       df = pd.Series([comment])
 #       features = self.preprocessor.createFeatureMatrix(df)
#
#        return self.model.predict(features)

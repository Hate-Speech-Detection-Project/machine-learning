from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from preprocessor import Preprocessor
from utils import AnalysisInformation
from classifier import Classifier

class RandomForestBOWClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.name = 'RandomForest'
        self.trained = False
        self.model = None
        self.testResult = None

    def fitFeatureMatrix(self, x, y):
        if not self.trained:
            self.model = RandomForestClassifier(n_estimators = 100)
            self.model.fit(x, y, sample_weight=self.getWeights(y))

            self.calibrated = CalibratedClassifierCV(self.model, method='isotonic', cv=10)
            self.calibrated.fit(x, y, sample_weight=self.getWeights(y))
            self.trained = True

    def predict(self, featureMatrix):
        return self.calibrated.predict_proba(featureMatrix)[:, 1][0]

 #   def predict(self, comment):
 #       df = pd.Series([comment])
 #       features = self.preprocessor.createFeatureMatrix(df)
#
#        return self.model.predict(features)

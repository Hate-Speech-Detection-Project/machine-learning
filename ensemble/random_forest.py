from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from preprocessor import Preprocessor
from utils import ConfusionMatrix

class RandomForestBOWClassifier:
    def __init__(self):
        self.model = None
        self.calibrated = None
        self.testResult = None

    def fitFeatureMatrix(self, x, y):

        self.model = RandomForestClassifier(n_estimators = 100)
        self.model.fit(x, y)

        self.calibrated = CalibratedClassifierCV(self.model, cv=2, method='isotonic')
        self.calibrated.fit(x, y)

    def testFeatureMatrix(self, x, y):
        if self.testResult == None:
            # Use the random forest to make sentiment label predictions
            result = self.model.predict(x)

            prob_pos_isotonic = self.calibrated.predict_proba(x)[:, 1]

            confusionMatrix = ConfusionMatrix(Preprocessor.convertBoolStringsToNumbers(result), Preprocessor.convertBoolStringsToNumbers(y))
            self.testResult = (confusionMatrix, result, prob_pos_isotonic)

        return self.testResult

 #   def predict(self, comment):
 #       df = pd.Series([comment])
 #       features = self.preprocessor.createFeatureMatrix(df)
#
#        return self.model.predict(features)

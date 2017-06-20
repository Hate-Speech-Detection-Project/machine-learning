from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from utils import ConfusionMatrix

from preprocessor import Preprocessor

class AdaBoost:
    def __init__(self, preprocessor):
        self.model = None
        self.preprocessor = preprocessor
        self.calibrated = None
        self.testResult = None

    def fit(self, train_df):
        trainingFeatures = self.preprocessor.trainFeatureMatrix(train_df);
        self.model = AdaBoostClassifier(n_estimators=100)
        self.model.fit(trainingFeatures, train_df['hate'])

        self.calibrated = CalibratedClassifierCV(self.model, cv=2, method='isotonic')
        self.calibrated.fit(trainingFeatures, train_df['hate'])

        print("done")

    def fitFeatureArray(self, x, y):
        clf = AdaBoostClassifier(n_estimators=100)
        self.model = clf.fit(x, y)

        self.calibrated = CalibratedClassifierCV(self.model, cv=2, method='isotonic')
        self.calibrated.fit(x, y)
        print("done")

    def test(self, test_df):

        if(self.testResult == None):
            test_data_features = self.preprocessor.createFeatureMatrix(test_df)
            test_data_features = test_data_features.toarray()

            # Use the random forest to make sentiment label predictions
            result = self.model.predict(test_data_features)

            prob_pos_isotonic = self.calibrated.predict_proba(test_data_features)[:, 1]

            confusionMatrix = ConfusionMatrix(Preprocessor.convertBoolStringsToNumbers(result), Preprocessor.convertBoolStringsToNumbers(test_df["hate"]))

            self.testResult = (confusionMatrix, result, prob_pos_isotonic)
        return self.testResult

    def testFeatuerMatrix(self, features, result):
        # Use the random forest to make sentiment label predictions
        predicted = self.model.predict(features)

        confusionMatrix = ConfusionMatrix(Preprocessor.convertBoolStringsToNumbers(predicted), Preprocessor.convertBoolStringsToNumbers(result))

        return (confusionMatrix, predicted)

    def predict(self, comment):
        df = pd.Series([comment])
        features = self.preprocessor.createFeatureMatrix(df)

        return self.model.predict(features)

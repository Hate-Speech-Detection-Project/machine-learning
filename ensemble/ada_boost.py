from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from utils import ConfusionMatrix

from preprocessor import Preprocessor

class AdaBoost:
    def __init__(self, preprocessor):
        self.model = None
        self.preprocessor = preprocessor

    def fit(self, train_df):
        trainingFeatures = self.preprocessor.trainFeatureMatrix(train_df);
        clf = AdaBoostClassifier(n_estimators=100)
        self.model = clf.fit(trainingFeatures, train_df['hate'])
        print("done")

    def fitFormatted(self, x, y):
        clf = AdaBoostClassifier(n_estimators=100)
        self.model = clf.fit(x, y)
        print("done")

    def test(self, test_df):
        test_data_features = self.preprocessor.createFeatureMatrix(test_df)
        test_data_features = test_data_features.toarray()

        # Use the random forest to make sentiment label predictions
        result = self.model.predict(test_data_features)
        confusionMatrix = ConfusionMatrix(Preprocessor.convertBoolStringsToNumbers(result), Preprocessor.convertBoolStringsToNumbers(test_df["hate"]))

        return (confusionMatrix, result)

    def testFeatuerMatrix(self, features, result):
        # Use the random forest to make sentiment label predictions
        predicted = self.model.predict(features)

        confusionMatrix = ConfusionMatrix(Preprocessor.convertBoolStringsToNumbers(predicted), Preprocessor.convertBoolStringsToNumbers(result))

        return (confusionMatrix, predicted)

    def predict(self, comment):
        df = pd.Series([comment])
        features = self.preprocessor.createFeatureMatrix(df)

        return self.model.predict(features)

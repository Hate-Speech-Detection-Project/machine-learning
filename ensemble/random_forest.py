from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from preprocessor import Preprocessor
from utils import ConfusionMatrix

class RandomForestBOWClassifier:
    def __init__(self, preprocessor):
        self.model = None
        self.preprocessor = preprocessor
        self.calibrated = None

    def fit(self, train_df):
        print("Training the random forest...")

        # Initialize a Random Forest classifier
        forest = RandomForestClassifier(n_estimators = 100)

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        trainingFeatures = self.preprocessor.trainFeatureMatrix(train_df);
        print(trainingFeatures)
        forest = forest.fit(trainingFeatures, train_df["hate"])
        self.model = forest

        self.calibrated = CalibratedClassifierCV(self.model, cv=2, method='isotonic')
        self.calibrated.fit(trainingFeatures, train_df["hate"])
        print("done")

    def fitFormatted(self, x, y):
        self.model = RandomForestClassifier(n_estimators = 100)
        self.model.fit(x, y)

        self.calibrated = CalibratedClassifierCV(self.model, cv=2, method='isotonic')
        self.calibrated.fit(x, y)
        print("done")


    def test(self, test_df):
        test_data_features = self.preprocessor.createFeatureMatrix(test_df)
        test_data_features = test_data_features.toarray()

        # Use the random forest to make sentiment label predictions
        result = self.model.predict(test_data_features)

        prob_pos_isotonic = self.calibrated.predict_proba(test_data_features)[:, 1]

        confusionMatrix = ConfusionMatrix(Preprocessor.convertBoolStringsToNumbers(result), Preprocessor.convertBoolStringsToNumbers(test_df["hate"]))
        return (confusionMatrix, result, prob_pos_isotonic)

    def predict(self, comment):
        df = pd.Series([comment])
        features = self.preprocessor.createFeatureMatrix(df)

        return self.model.predict(features)

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from preprocessor import Preprocessor
from utils import ConfusionMatrix

class RandomForestBOWClassifier:
    def __init__(self, preprocessor):
        self.model = None
        self.preprocessor = preprocessor

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
        print("done")

    def fitFormatted(self, x, y):
        forest = RandomForestClassifier(n_estimators = 100)
        forest = forest.fit(x, y)
        self.model = forest
        print("done")


    def test(self, test_df):
        test_data_features = self.preprocessor.createFeatureMatrix(test_df)
        test_data_features = test_data_features.toarray()

        # Use the random forest to make sentiment label predictions
        result = self.model.predict(test_data_features)

        confusionMatrix = ConfusionMatrix(Preprocessor.convertBoolStringsToNumbers(result), Preprocessor.convertBoolStringsToNumbers(test_df["hate"]))
        return (confusionMatrix, result)

    def predict(self, comment):
        df = pd.Series([comment])
        features = self.preprocessor.createFeatureMatrix(df)

        return self.model.predict(features)

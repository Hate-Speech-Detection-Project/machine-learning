from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from preprocessor import PreProcessor

class RandomForestBOWClassifier:
    def __init__(self):
        self.model = None
        self.preprocessor = PreProcessor()

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

        # Copy the results to a pandas dataframe with an "id" column and
        # a "sentiment" column
        output = pd.DataFrame(data={"cid": test_df["cid"], "hate": result})

        verification = pd.DataFrame(data={"cid": test_df["cid"], "hate": test_df["hate"]})
        merged = pd.merge(verification, output, on="cid")

        equal = merged.loc[merged["hate_x"] == merged["hate_y"]]
        acc = equal["cid"].count() / verification["cid"].count()

        print("Accuracy", acc)
        print("testresult:")
        return (acc, result)

    def predict(self, comment):
        df = pd.Series([comment])
        features = self.preprocessor.createFeatureMatrix(df)

        return self.model.predict(features)

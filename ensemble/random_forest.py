from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import preprocessor as pp

class RandomForestClassifier:
    def __init__(self):
        self.model = None
        pass

    def fit(self, train_df):
        print("Training the random forest...")

        # Initialize a Random Forest classifier
        forest = RandomForestClassifier()

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        trainingFeatures = pp.trainFeatureMatrix(train_df);
        print(trainingFeatures)
        self.model = forest.fit(trainingFeatures, train_df["hate"])
        print("done")


    def test(self, test_df):
        test_data_features = pp.createFeatureMatrix(test_df)
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
        return acc

    def predict(self, comment):
        df = pd.Series([comment])
        features = pp.createFeatureMatrix(df)

        return self.model.predict(features)
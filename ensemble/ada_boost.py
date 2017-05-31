from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

from preprocessor import PreProcessor

class AdaBoost:
    def __init__(self):
        self.model = None
        self.preprocessor = PreProcessor()

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

        # Copy the results to a pandas dataframe with an "id" column and
        # a "sentiment" column
        output = pd.DataFrame(data={"cid": test_df["cid"], "hate": result})

        verification = pd.DataFrame(data={"cid": test_df["cid"], "hate": test_df["hate"]})
        merged = pd.merge(verification, output, on="cid")

        equal = merged.loc[merged["hate_x"] == merged["hate_y"]]
        acc = equal["cid"].count() / verification["cid"].count()

        print("Accuracy", acc)
        return (acc, result)

    def predict(self, comment):
        df = pd.Series([comment])
        features = self.preprocessor.createFeatureMatrix(df)

        return self.model.predict(features)

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost:
    def __init__(self):
        self.model = None
        pass

    def fit(self, train_df):
        trainingFeatures = pp.trainFeatureMatrix(train_df);
        clf = AdaBoostClassifier(n_estimators=100)
        self.model = clf.train(trainingFeatures, train_df['hate'])
        scores.mean()
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

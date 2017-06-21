import numpy as np
from sklearn.svm import SVR
from article_features import ArticleFeatures
import datetime


class TextFeatureClassifier:
    def __init__(self):
        self.article_features = ArticleFeatures()
        self.train_df = None

    def calculate_features_with_dataframe(self, df):
        df["created"] = df["created"].astype("datetime64[ns]")

        hour = df["created"] % (24*60*60)
        total_length = df['comment'].apply(lambda x: len(x))
        num_of_words = df['comment'].apply(lambda x: len(x.split()))
        avg_length = df['comment'].apply(lambda x: np.average([len(a) for a in x.split()]))
        num_questions = df['comment'].apply(lambda x: x.count('?'))
        num_quote = df['comment'].apply(lambda x: x.count('"'))
        num_dot = df['comment'].apply(lambda x: x.count('.'))
        # num_repeated_dot = df['comment'].apply(lambda x: x.count('..'))
        num_exclamation = df['comment'].apply(lambda x: x.count('!'))
        # num_http = df['comment'].apply(lambda x: x.count('http'))
        ratio_capitalized = df['comment'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x))

        features = np.vstack((
          self.normalize(total_length), 
          self.normalize(num_questions), 
          self.normalize(num_exclamation), 
          self.normalize(num_of_words), 
          self.normalize(hour), 
          self.normalize(num_quote), 
          self.normalize(num_dot), 
          self.normalize(ratio_capitalized), 
          # self.normalize(num_http), 
          # self.normalize(num_repeated_dot), 
          self.normalize(avg_length)
        )).T
        return features

    def calculate_features(self,comment,timestamp):
        date = datetime.datetime.fromtimestamp(timestamp)
        total_length = len(comment)
        num_of_words = len(comment.split())
        num_questions = (comment.count('?'))
        num_exclamation = (comment.count('!'))
        features = np.vstack((total_length, num_questions, num_exclamation, num_of_words, date.hour)).T
        return features

    def calculate_time_feature(self, df):
        df["created"] = df["created"].astype("datetime64[ns]")
        minute = df.created.dt.minute
        return minute.values.reshape(-1, 1)

    def fit(self, train_df):
        self.X = self.calculate_features_with_dataframe(train_df)
        # self.X = self.calculate_time_feature(train_df)
        self.y = (train_df['hate'].replace('t', '1', regex=True)
                  .replace('f', '0', regex=True).astype(float))

        self.svr = SVR(kernel='rbf')
        self.model = self.svr.fit(self.X, self.y)

    def test(self, test_df):
        X = self.calculate_features_with_dataframe(test_df)
        # X = self.calculate_time_feature(test_df)
        y = (test_df['hate'].replace('t', '1', regex=True)
             .replace('f', '0', regex=True).astype(float))
        predicted = self.model.predict(X)

        acc = np.mean(np.round(predicted) == y)
        print("Accuracy", acc)
        return acc

    def predict(self, comment, timestamp):
        X_test = self.calculate_features(comment, timestamp)

        predicted = self.model.predict(X_test)
        return predicted

import numpy as np
import pandas as pd
from sklearn.svm import SVR

class TextFeatureClassifier:
  def __init__(self):
        self.train_df = None

  def calculate_features(self, df):
      df["created"] = df["created"].astype("datetime64[ns]")
      hour = df.created.dt.hour
      total_length = df['comment'].apply(lambda x: len(x))
      num_of_words = df['comment'].apply(lambda x: len(x.split()))
      num_questions = df['comment'].apply(lambda x: x.count('?'))
      num_exclamation = df['comment'].apply(lambda x: x.count('!'))
      features = np.vstack((total_length, num_questions, num_exclamation, num_of_words, hour)).T
      return features

  def calculate_time_feature(self, df):
      df["created"] = df["created"].astype("datetime64[ns]")
      minute = df.created.dt.minute
      return minute.values.reshape(-1, 1)

  def fit(self, train_df):
    self.X = self.calculate_features(train_df)
    # self.X = self.calculate_time_feature(train_df)
    self.y = (train_df['hate'].replace( 't','1', regex=True )
                              .replace( 'f','0', regex=True ).astype(float))

    self.svr = SVR(kernel='rbf')
    self.model = self.svr.fit(self.X, self.y)

  def test(self, test_df):
    X = self.calculate_features(test_df)
    # X = self.calculate_time_feature(test_df)
    y = (test_df['hate'].replace( 't','1', regex=True )
                        .replace( 'f','0',   regex=True ).astype(float))
    predicted = self.model.predict(X)

    acc = np.mean(np.round(predicted) == y)       
    print("Accuracy", acc)
    return acc

  def predict(self, comment_df):
    X_test = self.calculate_features(comment_df)

    predicted = self.model.predict(X_test)
    return predicted
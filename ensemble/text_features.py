import numpy as np
import pandas as pd
from sklearn.svm import SVR
from preprocessor import Preprocessor
from utils import ConfusionMatrix
from sklearn.calibration import CalibratedClassifierCV

class TextFeatureClassifier:
  def __init__(self):
        self.train_df = None
        self.calibrated = None
        self.testResult = None

  def calculate_features(self, df):
      total_length = df.apply(lambda x: len(x))
      num_of_words = df.apply(lambda x: len(x.split()))
      num_questions = df.apply(lambda x: x.count('?'))
      num_exclamation = df.apply(lambda x: x.count('!'))
      features = np.vstack((total_length, num_questions, num_exclamation, num_of_words)).T
      return features

  def fit(self, train_df):
    self.X = self.calculate_features(train_df['comment'])
    self.y = (train_df['hate'].replace( 't','1', regex=True )
                              .replace( 'f','0', regex=True ).astype(float))

    self.model = SVR(kernel='rbf')
    self.model.fit(self.X, self.y)

    # self.calibrated = CalibratedClassifierCV(self.model, cv=2, method='isotonic')
    # self.calibrated.fit(self.X, self.y)

  def test(self, test_df):

    if self.testResult == None:
      X = self.calculate_features(test_df['comment'])
      y = (test_df['hate'].replace( 't','1', regex=True )
                          .replace( 'f','0',   regex=True ).astype(float))
      predicted = self.model.predict(X)

      # prob_pos_isotonic = self.calibrated.predict_proba(X_new_tfidf)[:, 1]

      confusionMatrix = ConfusionMatrix(Preprocessor.convertBoolStringsToNumbers(predicted), Preprocessor.convertBoolStringsToNumbers(test_df['hate']))
      self.testResult = (confusionMatrix, predicted, [])

    return self.testResult

  def predict(self, comment):
    df = pd.Series([comment])
    X_test = self.calculate_features(df)

    predicted = self.model.predict(X_test)
    return predicted

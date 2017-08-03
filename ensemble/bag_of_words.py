from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from utils import AnalysisInformation
from preprocessor import Preprocessor
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from article_features import ArticleFeatures
import re
import nltk

class BagOfWordsClassifier:
  def __init__(self):
        self.trained = False
        self.tested = False
        self.train_df = None
        self.calibrated = None
        self.testResult = None
        self.feature_names = []
        self.hate_words_and_indices = []

  def fit(self, train_df):
    if not self.trained:
        self.train_df = train_df
        # Get training data
        X_train = self.train_df['comment']
        y_train = self.train_df['hate']

        # Tokenize Text
        from sklearn.feature_extraction.text import CountVectorizer
        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(X_train)
        X_train_counts.shape

        # From occurrences to frequencies
        from sklearn.feature_extraction.text import TfidfTransformer
        self.tfidf_transformer = TfidfTransformer()
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
        X_train_tfidf.shape

        # Training a classifier
        from sklearn.naive_bayes import MultinomialNB
        self.clf = MultinomialNB().fit(X_train_tfidf, y_train)
        self.hate_words = self.hate_words()

      # def fitFeatureMatrix(self, x, y):
      #     from sklearn.naive_bayes import MultinomialNB
      #     self.clf = MultinomialNB().fit(x, y)
      #     self.hate_words = self.hate_words()
      #     print("done")

        self.calibrated = CalibratedClassifierCV(self.clf, cv=2, method='isotonic')
        self.calibrated.fit(X_train_tfidf, y_train)
        self.trained = True

  def fitFeatureMatrix(self, x, y):
    if not self.trained:

        # Training a classifier
        from sklearn.naive_bayes import MultinomialNB
        self.clf = MultinomialNB().fit(x, y)
        self.calibrated = CalibratedClassifierCV(self.clf, cv=2, method='isotonic')
        self.calibrated.fit(x, y)
        self.trained = True

        rfe = RFE(self.clf, 3)
        fit = rfe.fit(x, y)
        print("Num Features: %d") % fit.n_features_
        print("Selected Features: %s") % fit.support_
        print("Feature Ranking: %s") % fit.ranking_


  def test(self, test_df):

    if not self.tested:
       # Get test data
      X_test = test_df['comment']
      y_test = test_df['hate']

      X_new_counts = self.count_vect.transform(X_test)
      X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
      predicted = self.clf.predict(X_new_tfidf)

      # acc = np.mean(predicted == y_test)
      prob_pos_isotonic = self.calibrated.predict_proba(X_new_tfidf)[:, 1]

      analysisInformation = AnalysisInformation(Preprocessor.convertBoolStringsToNumbers(predicted), Preprocessor.convertBoolStringsToNumbers(y_test))
      self.testResult = (analysisInformation, predicted, prob_pos_isotonic)
      self.tested = True

    return self.testResult

  def testFeatureMatrix(self, x, y):

    if not self.tested:
      predicted = self.clf.predict(x)

      prob_pos_isotonic = self.calibrated.predict_proba(x)[:, 1]

      analysisInformation = AnalysisInformation(Preprocessor.convertBoolStringsToNumbers(predicted), Preprocessor.convertBoolStringsToNumbers(y))
      self.testResult = (analysisInformation, predicted, prob_pos_isotonic)
      self.tested = True

    return self.testResult

  def predict(self, featureMatrix):
    predicted = self.calibrated.predict_proba(featureMatrix)[:, 1]
    if featureMatrix is None:
        return 0

    if len(self.feature_names) != 0:
        hate_words = [self.feature_names[i] for i in featureMatrix.nonzero()[1]]
        self.hate_words_and_indices = zip(featureMatrix.nonzero()[1], hate_words)

    return predicted[0]

  def predict_with_info(self, comment):
     # Get test data
    X_test = [comment]

    X_new_counts = self.count_vect.transform(X_test)
    X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
    predicted = self.clf.predict(X_new_tfidf)

    comment_clean = ''.join([x for x in comment if ord(x) < 128])
    words = comment_clean.lower().split()

    hate_series = pd.Series(self.hate_words)
    indexes = [hate_series[hate_series == word].index for word in words]
    indexes = [index[0] for index in indexes if len(index)>0]

    indices = np.argsort(indexes)
    sorted_index = np.asarray(indices)
    index_ordered_by_words = np.sort(indexes)
    hate_words = [words[val] + " (" + index_ordered_by_words[idx].encode('utf-8') + ")" for idx, val in enumerate(sorted_index)]

    return {
        "predicted": predicted,
        "hate_words": hate_words
    }

  def hate_words(self):
    # Top words
    X_train_hate = self.train_df[self.train_df['hate'] == True]['comment']
    X_train_no_hate = self.train_df[self.train_df['hate'] == False]['comment']

    X_train_hate_counts = self.count_vect.fit_transform(np.concatenate([X_train_hate, X_train_hate, X_train_no_hate]))
    X_train_hate_tfidf = self.tfidf_transformer.fit_transform(X_train_hate_counts)

    X_train_no_hate_counts = self.count_vect.fit_transform(np.concatenate([X_train_hate, X_train_no_hate, X_train_no_hate]))
    X_train_no_hate_tfidf = self.tfidf_transformer.fit_transform(X_train_no_hate_counts)

    words_mean_hate = X_train_hate_tfidf.mean(axis=0)
    words_mean_no_hate = X_train_no_hate_tfidf.mean(axis=0)

    substraction = np.subtract(words_mean_hate[0], words_mean_no_hate[0])
    indices = np.argsort(substraction)
    strings = self.count_vect.get_feature_names()
    index = np.asarray(indices)[0]
    hate_words = [strings[i] for i in reversed(index)]

    # print("Top 100 hate words", hate_words[:100])

    return hate_words

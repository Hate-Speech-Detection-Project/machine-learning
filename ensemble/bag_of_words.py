from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from utils import ConfusionMatrix
from preprocessor import Preprocessor
import pandas as pd
import numpy as np
from article_features import ArticleFeatures
import re
import nltk

class BagOfWordsClassifier:
  def __init__(self):
        self.train_df = None
        self.calibrated = None
        # self.article_features = ArticleFeatures()

  def fit(self, train_df):
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

  # def fitFormatted(self, x, y):
  #     from sklearn.naive_bayes import MultinomialNB
  #     self.clf = MultinomialNB().fit(x, y)
  #     self.hate_words = self.hate_words()
  #     print("done")

    self.calibrated = CalibratedClassifierCV(self.clf, cv=2, method='isotonic')
    self.calibrated.fit(X_train_tfidf, y_train)

  def test(self, test_df):
     # Get test data
    X_test = test_df['comment']
    y_test = test_df['hate']

    # X_test = self._remove_words_of_article_from_comments(test_df)

    X_new_counts = self.count_vect.transform(X_test)
    X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
    predicted = self.clf.predict(X_new_tfidf)

    # acc = np.mean(predicted == y_test)
    prob_pos_isotonic = self.calibrated.predict_proba(X_new_tfidf)[:, 1]

    confusionMatrix = ConfusionMatrix(Preprocessor.convertBoolStringsToNumbers(predicted), Preprocessor.convertBoolStringsToNumbers(y_test))
    return (confusionMatrix, predicted, prob_pos_isotonic)

  def predict(self, comment_df):
     # Get test data
    X_test = comment_df['comment']

    X_new_counts = self.count_vect.transform(X_test)
    X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
    predicted = self.clf.predict(X_new_tfidf)
    return predicted

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
    hate_words = [words[val] + " (" + str(index_ordered_by_words[idx]) + ")" for idx, val in enumerate(sorted_index)]

    return {
        "predicted": predicted,
        "hate_words": hate_words
    }

  # def _remove_words_of_article_from_comments(self,test_df):
  #   print('Removing specific comment-words...')
  #   index = 0
  #   X_test = test_df['comment']
  #   cids_from_comments = test_df['cid']
  #   for cid in cids_from_comments:
  #     intersection = self.article_features.get_shared_words_from_comment_and_article_by_cid(int(cid))
  #     text = X_test[index]
  #     X_test.loc[index] = ' '.join([w for w in nltk.word_tokenize(text) if not w.lower() in intersection])
  #     index += 1
  #   return X_test

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

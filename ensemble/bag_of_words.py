from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from article_features import ArticleFeatures
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

class BagOfWordsClassifier:
  def __init__(self):
    self.train_df = None

    # initialize stemmer
    self.stemmer = SnowballStemmer("german")
    # grab stopword list
    self.stop = stopwords.words('german')

  def text_to_wordlist(self, comment):
      try:
          comment_text = re.sub(r'https?:\/\/.*[\r\n]*', '', comment, flags=re.MULTILINE)
          comment_text = re.sub(r'<\/?em>', '', comment_text, flags=re.MULTILINE)
          comment_text = re.sub("[^a-zA-ZöÖüÜäÄß]"," ", comment_text)
          comment_text = re.sub("\s\s+"," ", comment_text)
          comment_text = comment_text.lower() + '. '
      #comment_text = unicode(comment_text, errors='replace')
      except:
          comment_text = ''
      return comment_text

  def to_wordlist(self, data):
      return data.apply(self.text_to_wordlist)

  def remove_stopwords(self, data):
      return data.apply(lambda x: [item for item in str(x).split(' ') if item not in self.stop])

  def stem(self, data):
      return data.apply(lambda x: " ".join([self.stemmer.stem(y) for y in x]))

  def remove_stop_and_stem(self, data):
      data = self.remove_stopwords(data)
      data = self.stem(data)
      return data

  def fit(self, train_df):
    self.train_df = train_df
    # Get training data
    X_train = self.train_df['comment']
    y_train = self.train_df['hate']

    # print(X_train)
    hans = self.remove_stop_and_stem(X_train)
    # print(X_train.shape, hans.shape)
    for i in range(0, len(X_train)):
      X_train.iloc[i] = hans.iloc[i]

    print(X_train.head())

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

  def test(self, test_df):
     # Get test data
    X_test = test_df['comment']
    y_test = test_df['hate']

    X_new_counts = self.count_vect.transform(X_test)
    X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
    predicted = self.clf.predict(X_new_tfidf)

    acc = np.mean(predicted == y_test)
    return acc

  def predict(self, comment_df):
     # Get test data
    X_test = comment_df['comment']
    X_test = self.remove_stop_and_stem(X_test)

    X_new_counts = self.count_vect.transform(X_test)
    X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
    print(X_new_tfidf[0])
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


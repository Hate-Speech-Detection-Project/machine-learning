import numpy as np
import pandas as pd
from sklearn.svm import SVR
import datetime
from gensim.models import Word2Vec
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import multiprocessing
from multiprocessing import cpu_count, Pool
from joblib import Parallel
import re
from itertools import chain, repeat, islice

class Doc2VecClassifier:
    def __init__(self):
        self.train_df = None
        self.d2v_model = Word2Vec.load('model/doc2vec')
        # initialize stemmer
        self.stemmer = SnowballStemmer("german")
        # grab stopword list
        self.stop = stopwords.words('german')
        self.hate_model = Word2Vec.load('softmax_models/w2v_hate')
        self.no_hate_model = Word2Vec.load('softmax_models/w2v_no_hate')

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
        data = self.to_wordlist(data)
        data = self.remove_stopwords(data)
        data = self.stem(data)
        return data

    def calculate_features_with_dataframe(self, test_df):
        data = self.remove_stop_and_stem(test_df['comment'])
        # TODO: Sentences are already lost here!
        data = data.apply(lambda x: [sentence.split(' ') for sentence in x.split('.')])
        data = data.apply(lambda x: self.d2v_model.score(x))

    def fit(self, train_df):
        self.X = self.calculate_features_with_dataframe(train_df)
        self.y = train_df['hate']
        if isinstance(self.y.iloc[0], str):
            self.y = self.y == 'True'
        self.svr = SVR(kernel='rbf')
        self.model = self.svr.fit(self.X, self.y)

    def test(self, test_df):
        data = self.remove_stop_and_stem(test_df['comment'])
        # TODO: Sentences are already lost here!
        data = data.apply(lambda x: [sentence.split(' ') for sentence in x.split('.')])
        data = data.apply(lambda x: self.d2v_model.score(x))
        return data

    def predict(self, comment):
        data = pd.Series([comment], index=['comment'])
        data = self.remove_stop_and_stem(df['comment'])
        data = data.apply(lambda x: self.d2v_model.score([sentence.split(' ') for sentence in x.split(' ')]))
        print(data)
        # predicted = self.d2v_model.score()
        return predicted

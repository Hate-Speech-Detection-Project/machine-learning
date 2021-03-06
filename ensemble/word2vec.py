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
from itertools import chain, repeat, islice


class Word2VecClassifier:
    def __init__(self):
        self.train_df = None
        self.model = Word2Vec.load('model/modelw2v_model')
        # initialize stemmer
        self.stemmer = SnowballStemmer("german")
        # grab stopword list
        self.stop = stopwords.words('german')

    def parallelize(self, data, func):
        cores = 3 #cpu_count() #Number of CPU cores on your system
        partitions = cores #Define as many partitions as you want
        data_split = np.array_split(data, partitions)
        pool = Pool(cores)
        data = pd.concat(pool.map(func, data_split))
        pool.close()
        pool.join()
        return data

    def pad_infinite(self, iterable, padding=None):
       return chain(iterable, repeat(padding))

    def pad(self, iterable, size, padding=None):
       return islice(self.pad_infinite(iterable, padding), size)

    def text_to_wordlist(self, comment):
        try:
            comment_text = re.sub(r'https?:\/\/.*[\r\n]*', '', comment, flags=re.MULTILINE)
            comment_text = re.sub(r'<\/?em>', '', comment_text, flags=re.MULTILINE)
            comment_text = re.sub("[^a-zA-ZöÖüÜäÄß]"," ", comment_text)
            comment_text = re.sub("\s\s+"," ", comment_text)
            comment_text = comment_text.lower() + '. '
            comment_text = unicode(comment_text, errors='replace')
        except:
            comment_text = ''
        return comment_text

    def to_wordlist(self, data):
        return data.apply(self.text_to_wordlist)

    def remove_stopwords(self, data):
        return data.apply(lambda x: [item for item in str(x).split(' ') if item not in self.stop])

    def stem(self, data):
        return data.apply(lambda x: " ".join([self.stemmer.stem(y) for y in x]))

    def word_to_position(self, word):
        try:
            return np.average(self.model.wv[word])
        except:
            return 0

    def comment_to_vectors(self, comment):
        words = comment.split(' ')
        # print("Comment:", words)
        result = list(map(self.word_to_position, words))
        # print("Result:", result)
        result = list(self.pad(result, 500, np.average(result)))
        # print(result)
        return result

    def remove_stop_and_stem(self, data):
        print("To Wordlist")
        data = self.parallelize(data, self.to_wordlist)
        print("Remove Stopwords")
        data = self.parallelize(data, self.remove_stopwords)
        print("Stem")
        data = self.parallelize(data, self.stem)
        return data

    def calculate_features_with_dataframe(self, df):
        data = self.remove_stop_and_stem(df['comment'])
        print("Calculate Word2Vec")
        vectors = list(map(self.comment_to_vectors, data))

        features = np.vstack((
          vectors, 
        )).T
        return vectors

    def fit(self, train_df):
        self.X = self.calculate_features_with_dataframe(train_df)
        self.y = train_df['hate']
        self.svr = SVR(kernel='rbf')
        self.model = self.svr.fit(self.X, self.y)

    def test(self, test_df):
        X = self.calculate_features_with_dataframe(test_df)
        y = test_df['hate']
        predicted = self.model.predict(X)
        return predicted

    def predict(self, comment):
        X_test = self.calculate_features_with_dataframe(pd.Series([comment], index=['comment'])).iloc[0]

        predicted = self.model.predict(X_test)
        return predicted

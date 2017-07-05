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
import types

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class Word2VecEnsembleClassifier:
    def __init__(self):
        self.train_df = None
        self.w2v_model_articles_lowercased = Word2Vec.load('model/articles_lowercased')
        self.w2v_model_comments_lowercased = Word2Vec.load('model/comments_lowercased')
        self.w2v_model_text_lowercased = Word2Vec.load('model/text_lowercased')
        self.w2v_model_w2v_stemmed_nostop = Word2Vec.load('model/w2v_stemmed_nostop')
        # initialize stemmer
        self.stemmer = SnowballStemmer("german")
        # grab stopword list
        self.stop = stopwords.words('german')

        
        self.count_vect = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()

    def parallelize(self, data, func):
        cores = 3 #cpu_count() #Number of CPU cores on your system
        partitions = cores #Define as many partitions as you want
        data_split = np.array_split(data, partitions)
        pool = Pool(cores)
        data = pd.concat(pool.map(func, data_split))
        pool.close()
        pool.join()
        return data

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

    def word_to_position(self, model, word):
        try:
            return model.wv[word]
        except:
            return -1

    def comment_to_vectors(self, comment, comment_stemmed):
        words = comment.split(' ')
        words_stemmed = comment_stemmed.split(' ')
        # result = list(map(self.word_to_position, words))
        result_articles_lowercased = list(map(lambda x: self.word_to_position(self.w2v_model_articles_lowercased, x), words))
        result_comments_lowercased = list(map(lambda x: self.word_to_position(self.w2v_model_comments_lowercased, x), words))
        result_text_lowercased = list(map(lambda x: self.word_to_position(self.w2v_model_text_lowercased, x), words))
        result_w2v_stemmed_nostop = list(map(lambda x: self.word_to_position(self.w2v_model_w2v_stemmed_nostop, x), words_stemmed))
        # print("Result:", result)

        # Tokenize Text
        tf = self.tfidf_transformer.transform(self.count_vect.transform([comment]))
        # print(comment_tfidf)

        feature_names = self.tfidf_transformer.get_feature_names()

        doc = 0
        feature_index = tfidf_matrix[doc,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])

        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            print (w, s)

        print(comment)
        print(comment_tfidf.toarray())
        # print(comment_tfidf[0])

        result_articles_lowercased = sum([a*b for a,b in zip(comment_tfidf[0],result_articles_lowercased)]) / len(words)
        result_comments_lowercased = sum([a*b for a,b in zip(comment_tfidf[0],result_comments_lowercased)]) / len(words)
        result_text_lowercased = sum([a*b for a,b in zip(comment_tfidf[0],result_text_lowercased)]) / len(words)
        result_w2v_stemmed_nostop = sum([a*b for a,b in zip(comment_tfidf[0],result_w2v_stemmed_nostop)]) / len(words_stemmed)
        # print(result)
        # return [result_articles_lowercased, result_comments_lowercased, result_text_lowercased, result_w2v_stemmed_nostop]
        return [result_w2v_stemmed_nostop]

    def remove_stop_and_stem(self, data):
        # print("To Wordlist")
        # data = self.parallelize(data, self.to_wordlist)
        # print("Remove Stopwords")
        # data = self.parallelize(data, self.remove_stopwords)
        # print("Stem")
        # data = self.parallelize(data, self.stem)
        # data = self.to_wordlist(data)
        data = self.remove_stopwords(data)
        data = self.stem(data)
        return data

    def lowercase_comments(self, data):
        return data

    def calculate_features_with_dataframe(self, df):
        data = self.to_wordlist(df['comment'])
        stemmed_data = self.remove_stop_and_stem(data)
        # print(data)
        vectors = list(map(self.comment_to_vectors, data, stemmed_data))
        # print(len(vectors))
        # features = list(map(lambda x: np.concatenate((x[0], x[1], x[2], x[3]), axis=0), vectors))
        features = list(map(lambda x: x[0], vectors))
        # print(features.shape)
        return features

    def fit(self, train_df):
        data = self.to_wordlist(train_df['comment'])
        # print("DATA", data)
        # # Tokenize Text
        self.X_train_counts = self.count_vect.fit_transform(data)
        # # From occurrences to frequencies
        self.X_train_tfidf = self.tfidf_transformer.fit(self.X_train_counts)

        self.X = self.calculate_features_with_dataframe(train_df)
        self.y = train_df['hate']
        if isinstance(self.y.iloc[0], str):
            self.y = self.y == 'True'
        self.svr = SVR(kernel='rbf')
        self.model = self.svr.fit(self.X, self.y)

    def test(self, test_df):
        print("="*20)
        X = self.calculate_features_with_dataframe(test_df)
        y = test_df['hate']
        if isinstance(y.iloc[0], str):
            y = y == 'True'
        predicted = self.model.predict(X)
        return predicted

    def predict(self, comment):
        X_test = self.calculate_features_with_dataframe(pd.Series([comment], index=['comment'])).iloc[0]

        predicted = self.model.predict(X_test)
        return predicted

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

class Word2VecDeepInverseRegressionClassifier:
    def __init__(self):
        self.train_df = None
        # initialize stemmer
        self.stemmer = SnowballStemmer("german")
        # grab stopword list
        self.stop = stopwords.words('german')
        self.hate_model = Word2Vec.load('softmax_models/w2v_hate')
        self.no_hate_model = Word2Vec.load('softmax_models/w2v_no_hate')

    def docprob(self, docs, mods):
        # score() takes a list [s] of sentences here; could also be a sentence generator
        sentlist = [s for d in docs for s in d]
        # the log likelihood of each sentence in this comment under each w2v representation
        llhd = np.array( [ m.score(sentlist, len(sentlist)) for m in mods ] )
        # now exponentiate to get likelihoods, 
        lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
        # normalize across models (stars) to get sentence-star probabilities
        prob = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose() )
        # and finally average the sentence probabilities to get the comment probability
        prob["doc"] = [i for i,d in enumerate(docs) for s in d]
        prob = prob.groupby("doc").mean()
        return prob

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

    def remove_stop_and_stem(self, data):
        # print("To Wordlist")
        # data = self.parallelize(data, self.to_wordlist)
        # print("Remove Stopwords")
        # data = self.parallelize(data, self.remove_stopwords)
        # print("Stem")
        # data = self.parallelize(data, self.stem)
        data = self.to_wordlist(data)
        data = self.remove_stopwords(data)
        data = self.stem(data)
        return data

    def fit(self, train_df):
        pass

    def test(self, test_df):
        data = self.remove_stop_and_stem(test_df['comment'])
        # get the probs (note we give docprob a list of lists of words, plus the models)
        # print(self.docprob( data.apply(lambda x: str(x).split(' ')), [self.no_hate_model, self.hate_model] ))
        predicted = self.docprob( data.apply(lambda x: str(x).split(' ')), [self.no_hate_model, self.hate_model])[0]
        return predicted

    def predict(self, comment):
        data = pd.Series([comment], index=['comment'])
        data = self.remove_stop_and_stem(df['comment'])
        predicted = self.docprob(str(data.iloc[0]).split(' ')), [self.no_hate_model, self.hate_model][0]
        return predicted

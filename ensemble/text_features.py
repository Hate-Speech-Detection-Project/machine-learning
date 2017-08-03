from textblob_de import TextBlobDE as TextBlob
from textblob_de import PatternParser
from textblob_de import PatternTagger
from topic_features import TopicFeatures
import numpy as np
from sklearn.svm import SVR
import datetime
from textblob_de import TextBlobDE as TextBlob
from textblob_de import PatternParser
from threading import Thread
from enum import Enum
import pandas as pd
import time
import scipy.stats
import matplotlib.pyplot as plt

RESULT_COUNT = 12


class Resultindices(Enum):
    LENGTH_OF_COMMENT, NUM_OF_WORDS, NUM_OF_DISTINCT_WORDS, \
    NUM_OF_QUESTIONMARKS, NUM_OF_EXCLAMATIONMARKS, NUM_OF_ADJECTIVES, \
    NUM_OF_DETERMINER, NUM_OF_PERSONAL_PRONOUNS, NUM_OF_ADVERBS, \
    NUM_OF_INTERJECTIONS, SUBJECTIVITY_VALUE, POLARITY_VALUE = range(
        RESULT_COUNT)


class TextFeatureGenerator:
    def __init__(self):
        self.train_df = None
        self.topic_features = TopicFeatures()
        self.results = [0] * RESULT_COUNT

    def tagCommentsFromDf(self, df):
        return df.apply(lambda x: TextBlob(x).tags)

    def calculate_features_with_dataframe(self, df):

        threads = []
        df["created"] = df["created"].astype("datetime64[ns]")

        hour = df.created.dt.hour

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: len(x)))),
                args=(df, self.results, Resultindices.LENGTH_OF_COMMENT.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: len(x.split())))),
                args=(df, self.results, Resultindices.NUM_OF_WORDS.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: len(set(x.split()))))),
                args=(df, self.results, Resultindices.NUM_OF_DISTINCT_WORDS.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: x.count('?')))),
                args=(df, self.results, Resultindices.NUM_OF_QUESTIONMARKS.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: x.count('!')))),
                args=(df, self.results, Resultindices.NUM_OF_EXCLAMATIONMARKS.value)))

        # semantic analysis
        self._start_threads_and_join(threads)

        print('Start tagging for semantic analysis')
        threads = []
        text_blob_comments = df['comment'].apply(lambda x: TextBlob(x))
        tagged_comments = text_blob_comments.apply(lambda x: x.tags)
        print('finished tagging')

        threads.append(
            Thread(target=(
                lambda x, results, index: results.insert(index, tagged_comments.apply(
                    lambda x: TextFeatureGenerator._getCountOfWordsByTaggedList(x, ['JJ', 'JJS', 'JJR'])))),
                args=(tagged_comments, self.results, Resultindices.NUM_OF_ADJECTIVES.value)))

        threads.append(
            Thread(target=(
                lambda x, results, index: results.insert(index, tagged_comments.apply(
                    lambda x: TextFeatureGenerator._getCountOfWordsByTaggedList(x, ['DT'])))),
                args=(tagged_comments, self.results, Resultindices.NUM_OF_DETERMINER.value)))

        threads.append(
            Thread(target=(
                lambda x, results, index: results.insert(index, tagged_comments.apply(
                    lambda x: TextFeatureGenerator._getCountOfWordsByTaggedList(x, ['PRP'])))),
                args=(tagged_comments, self.results, Resultindices.NUM_OF_PERSONAL_PRONOUNS.value)))

        threads.append(
            Thread(target=(
                lambda x, results, index: results.insert(index, tagged_comments.apply(
                    lambda x: TextFeatureGenerator._getCountOfWordsByTaggedList(x, ['RB', 'RBS'])))),
                args=(tagged_comments, self.results, Resultindices.NUM_OF_ADVERBS.value)))

        threads.append(
            Thread(target=(
                lambda x, results, index: results.insert(index, tagged_comments.apply(
                    lambda x: TextFeatureGenerator._getCountOfWordsByTaggedList(x, ['UH'])))),
                args=(tagged_comments, self.results, Resultindices.NUM_OF_INTERJECTIONS.value)))

        # # calculation of cosinent similarity for relation between comment/article/hate+no-hate comments of the article
        print('Start semantic analysis')
        self._start_threads_and_join(threads)
        print('Finished semantic analysis')

        threads = []

        print('Start calculation of cosinent-similarity for semantic analysis')
        cos_similarity_article = []
        cos_similarity_no_hate_comments = []
        cos_similarity_hate_comments = []

        threads.append(Thread(target=self._calculate_article_cos_similarity,
                              args=(df, cos_similarity_article, self.topic_features)))

        threads.append(Thread(target=self._calculate_no_hate_comments_cos_similarity,
                              args=(df, cos_similarity_no_hate_comments, self.topic_features)))

        threads.append(Thread(target=self._calculate_hate_cos_similarity,
                              args=(df, cos_similarity_hate_comments, self.topic_features)))

        self._start_threads_and_join(threads)

        features = np.vstack((
            self.results[Resultindices.NUM_OF_EXCLAMATIONMARKS.value],
            self.results[Resultindices.NUM_OF_QUESTIONMARKS.value],
            self.results[Resultindices.NUM_OF_DISTINCT_WORDS.value],
            self.results[Resultindices.NUM_OF_WORDS.value],
            self.results[Resultindices.LENGTH_OF_COMMENT.value],

            self.results[Resultindices.NUM_OF_INTERJECTIONS.value],
            self.results[Resultindices.NUM_OF_ADVERBS.value],
            self.results[Resultindices.NUM_OF_PERSONAL_PRONOUNS.value],
            self.results[Resultindices.NUM_OF_DETERMINER.value],
            self.results[Resultindices.NUM_OF_ADJECTIVES.value],

            cos_similarity_article,
            cos_similarity_no_hate_comments,
            cos_similarity_hate_comments,
            hour
        )).T

        data = np.corrcoef(features)
        print(np.corrcoef(features))

        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data)

        # put the major ticks at the middle of each cell, notice "reverse" use of dimension
        ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)

        plt.show()

        return features

    def _start_threads_and_join(self, threads):
        for thread in threads:
            thread.start()
            thread.join()


    def _calculate_article_cos_similarity(self, df, cos_list, topic_features):
        for index, row in df.iterrows():
            cos_list.append(topic_features.get_cos_similarity_for_article(row['comment'], row['url']))

    def _calculate_hate_cos_similarity(self, df, cos_list, topic_features):
        for index, row in df.iterrows():
            cos_list.append(topic_features.get_cos_similarity_for_hate_comments_of_article(row['comment'], row['url']))

    def _calculate_no_hate_comments_cos_similarity(self, df, cos_list, topic_features):
        for index, row in df.iterrows():
            cos_list.append(
                topic_features.get_cos_similarity_for_no_hate_comments_of_article(row['comment'], row['url']))

    def calculate_features(self, comment, timestamp):
        tagged_comment = TextBlob(comment).tags
        date = datetime.datetime.fromtimestamp(timestamp)
        total_length = len(comment)
        num_of_words = len(comment.split())
        num_questions = (comment.count('?'))
        num_exclamation = (comment.count('!'))
        num_adjectives = TextFeatureGenerator._getCountOfWordsByTaggedList(tagged_comment, ['JJ', 'JJS', 'JJR'])
        num_determiner = TextFeatureGenerator._getCountOfWordsByTaggedList(tagged_comment, ['DT'])
        num_personal_pronouns = TextFeatureGenerator._getCountOfWordsByTaggedList(tagged_comment, ['PRP'])
        num_interjections = TextFeatureGenerator._getCountOfWordsByTaggedList(tagged_comment, ['UH'])
        num_adverbs = TextFeatureGenerator._getCountOfWordsByTaggedList(tagged_comment, ['RB', 'RBS'])
        features = np.vstack((total_length, num_questions, num_exclamation, num_of_words,
                              date.hour, num_adjectives, num_determiner, num_personal_pronouns, num_adverbs,
                              num_interjections)).T
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
        return acc

    def predict(self, comment, timestamp):
        X_test = self.calculate_features(comment, timestamp)

        predicted = self.model.predict(X_test)
        return predicted

    @staticmethod
    def _getCountOfWordsByTaggedList(tagged_list, tag_id_list):
        count = 0
        for tag in tagged_list:
            if tag[1] in tag_id_list:
                count = count + 1

        return count

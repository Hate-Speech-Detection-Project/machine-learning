from textblob_de import TextBlobDE as TextBlob
from textblob_de import PatternParser
from textblob_de import PatternTagger
import numpy as np
from sklearn.svm import SVR
from article_features import ArticleFeatures
import datetime
from textblob_de import TextBlobDE as TextBlob
from textblob_de import PatternParser


class TextFeatureClassifier:
    def __init__(self):
        self.article_features = ArticleFeatures()
        self.train_df = None

    def calculate_features_with_dataframe(self, df):
        tagged_comments = df['comment'].apply(lambda x: TextBlob(x).tags)

        df["created"] = df["created"].astype("datetime64[ns]")
        hour = df.created.dt.hour
        total_length = df['comment'].apply(lambda x: len(x))
        num_of_words = df['comment'].apply(lambda x: len(x.split()))
        num_of_distinct_words = df['comment'].apply(lambda x: len(set(x.split())))
        num_questions = df['comment'].apply(lambda x: x.count('?'))
        num_exclamation = df['comment'].apply(lambda x: x.count('!'))
        num_adjectives = tagged_comments.apply(lambda x: TextFeatureClassifier._getCountOfWordsByTaggedList(x, ['JJ', 'JJS', 'JJR']))
        num_determiner = tagged_comments.apply(lambda x: TextFeatureClassifier._getCountOfWordsByTaggedList(x, ['DT']))
        num_personal_pronouns = tagged_comments.apply(lambda x: TextFeatureClassifier._getCountOfWordsByTaggedList(x, ['PRP']))
        num_adverbs = tagged_comments.apply(lambda x: TextFeatureClassifier._getCountOfWordsByTaggedList(x, ['RB', 'RBS']))
        num_interjections = tagged_comments.apply(lambda x: TextFeatureClassifier._getCountOfWordsByTaggedList(x, ['UH']))

        # TODO calculates the sentiment values for each comment, nevertheless it is not worth the effort
        # sentiment_analysis = df['comment'].apply(lambda x: (TextBlob(x, parser=PatternParser(pprint=True, lemmata=True))).sentiment[0])

        features = np.vstack(
            (total_length, num_questions, num_exclamation, num_of_words, hour,
             num_of_distinct_words, num_adjectives, num_determiner, num_personal_pronouns, num_adverbs,num_interjections)).T
        return features

    def calculate_features(self, comment, timestamp):
        tagged_comment = TextBlob(comment).tags

        date = datetime.datetime.fromtimestamp(timestamp)
        total_length = len(comment)
        num_of_words = len(comment.split())
        num_questions = (comment.count('?'))
        num_exclamation = (comment.count('!'))
        num_adjectives = TextFeatureClassifier._getCountOfWordsByTaggedList(tagged_comment, ['JJ', 'JJS', 'JJR'])
        num_superlatives = TextFeatureClassifier._getCountOfWordsByTaggedList(tagged_comment, ['DT'])
        num_personal_pronouns = TextFeatureClassifier._getCountOfWordsByTaggedList(tagged_comment, ['PRP'])
        num_interjections = TextFeatureClassifier._getCountOfWordsByTaggedList(tagged_comment, ['UH'])
        num_adverbs = TextFeatureClassifier._getCountOfWordsByTaggedList(tagged_comment, ['RB', 'RBS'])
        features = np.vstack((total_length, num_questions, num_exclamation, num_of_words,
                              date.hour, num_adjectives, num_superlatives, num_personal_pronouns,num_adverbs,num_interjections)).T
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

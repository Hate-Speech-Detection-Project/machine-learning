import psycopg2
import sys
import numpy as np
from nltk.corpus import stopwords  # Import the stop word list
import nltk


class ArticleFeatures:
    def __init__(self):
        # set the stopwords for german
        self.stops = []
        self.read_in_stopword()

        try:
            self.conn = psycopg2.connect("dbname='hatespeech' user='postgres' host='localhost' password='admin'")
        except:
            print("Cannot connect to database.")
            sys.exit(0)

    def read_in_stopword(self):
        file = open('../data/stopwords_de.txt', 'rU', encoding="utf8")
        for line in file.readlines():
            self.stops.append(line.replace('\n', '').replace('\r', ''))

    def remove_stopwords(self, text):
        # we delete the '.' and ',' because they are not set properly
        text = text.replace('.', ' ').replace(',', ' ')

        tokenized_text = nltk.word_tokenize(text)
        return ' '.join([w for w in tokenized_text if not w in self.stops])

    def get_shared_words_from_comment_and_article_by_cid(self, cid):
        cur = self.conn.cursor()

        query = """ SELECT body FROM articles WHERE url = ( SELECT url FROM comments WHERE cid = %s ) """
        cur.execute(query, [cid])

        result = cur.fetchone()

        if result is None:
            print('Could not find article for comment with cid:' + str(cid))
            return []

        article_body = result[0]
        article_body = self.remove_stopwords(article_body.lower())

        query = """ SELECT comment FROM comments WHERE cid = %s """
        cur.execute(query, [cid])

        comment = self.remove_stopwords(cur.fetchone()[0].lower())

        # Tokenize Text
        from sklearn.feature_extraction.text import CountVectorizer

        count_vect = CountVectorizer()

        # count the word occurrences from the article
        count_vect.fit_transform([article_body])
        article_word_list = np.array(count_vect.get_feature_names())

        count_vect_comment = CountVectorizer()
        # count the word occurrences in the comments
        count_vect_comment.fit_transform([comment])
        comment_word_list = np.array(count_vect_comment.get_feature_names())

        intersection = []
        for word in comment_word_list:
            if word in article_word_list:
                intersection.append(word)
        return intersection

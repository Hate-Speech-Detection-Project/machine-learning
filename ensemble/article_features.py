import psycopg2
import sys
import numpy as np
from nltk.corpus import stopwords  # Import the stop word list
import nltk


class ArticleFeatures:
    def __init__(self):
        # set the stopwords for german
        self.stops = []
        self._read_in_stopword()
        self.conn = None

    def connect_to_database(self):
        if self.conn is None:
            try:
                # self.conn = psycopg2.connect("dbname='hatespeech' user='postgres' host='localhost' password='admin'")
                self.conn = psycopg2.connect(
                    "dbname='hatespeech' user='postgres' password='admin' port='5432' host='172.17.0.2'")
            except:
                print("Cannot connect to database.")
                sys.exit(0)


    def _read_in_stopword(self):
        file = open('../data/stopwords_de.txt', 'rU', encoding="utf8")
        for line in file.readlines():
            self.stops.append(line.replace('\n', '').replace('\r', ''))

    def get_shared_words_from_comment_and_article_using_db_by_cid(self, cid):
        self.connect_to_database()

        cur = self.conn.cursor()

        query = """ SELECT body FROM articles WHERE url = ( SELECT url FROM comments WHERE cid = %s ) """
        cur.execute(query, [cid])

        result = cur.fetchone()
        if result is None:
            print('Could not find article for comment with cid: ' + str(cid))
            return []

        article_body = result[0]
        article_body = self.remove_stopwords(article_body.lower())

        query = """ SELECT comment FROM comments WHERE cid = %s """
        cur.execute(query, [cid])

        comment = self.remove_stopwords(cur.fetchone()[0].lower())

        if not self._has_content(comment) or not self._has_content(article_body):
            print('Comment or article_body to short to find something for cid: ' + str(cid))
            return []

        return ArticleFeatures.get_word_intersection_from_texts(article_body, comment)


    def remove_words_of_article_from_comments_by_dataframe(self, test_df):
        print('Removing specific comment-words...')
        index = 0
        X_test = test_df['comment']
        cids_from_comments = test_df['cid']
        for cid in cids_from_comments:
            if (index % 100 == 0):
                print('---------------- Cleaned ' + str(index) + ' Comments --------------------')
            intersection = self.get_shared_words_from_comment_and_article_using_db_by_cid(int(cid))
            text = X_test[index]
            test_df.loc[index, 'comment'] = ' '.join(
                [w for w in nltk.word_tokenize(text) if not w.lower() in intersection])
            index += 1
        return test_df

    def _has_content(self, text):
        if text == " " or len(text) <= 1:
            return False
        return True


    def remove_words_of_comment_by_given_text(self,comment, text):
        cleaned_text = self.remove_stopwords(text.lower())
        cleaned_comment =  self.remove_stopwords(comment.lower())
        intersection = ArticleFeatures.get_word_intersection_from_texts(cleaned_comment,cleaned_text)
        return ' '.join([w for w in nltk.word_tokenize(cleaned_comment) if not w.lower() in intersection])

    def remove_stopwords(self,text):
        # we delete the '.' and ',' because they are not set properly
        text = text.replace('.', ' ').replace(',', ' ')

        tokenized_text = nltk.word_tokenize(text)

        return ' '.join([w for w in tokenized_text if not w in self.stops])


    # ************************************** static methods ************************************

    @staticmethod
    def get_word_intersection_from_texts(text1, text2):
        # Tokenize Text
        from sklearn.feature_extraction.text import CountVectorizer

        count_vect = CountVectorizer()

        # count the word occurrences from the article
        count_vect.fit_transform([text1])
        article_word_list = np.array(count_vect.get_feature_names())

        count_vect_comment = CountVectorizer()
        # count the word occurrences in the comments

        count_vect_comment.fit_transform([text2])
        comment_word_list = np.array(count_vect_comment.get_feature_names())

        intersection = []
        for word in comment_word_list:
            if word in article_word_list:
                intersection.append(word)
        return intersection

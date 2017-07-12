from __future__ import division
import psycopg2
import sys
import numpy as np


class UserFeatureGenerator:
    def calculate_features_with_dataframe(self, df):
        time_since_last_comment_by_user = []
        time_since_last_hate_comment_by_user = []
        number_of_comments_by_user = []
        number_of_hate_comments_by_user = []
        share_of_hate_comments_by_user = []

        for index, comment in df.iterrows():
            time_since_last_comment_by_user.append(self.time_since_last_comment_by_user(comment['uid']))
            time_since_last_hate_comment_by_user.append(self.time_since_last_hate_comment_by_user(comment['uid']))
            number_of_comments_by_user.append(self.number_of_comments_by_user(comment['uid']))
            number_of_hate_comments_by_user.append(self.number_of_hate_comments_by_user(comment['uid']))
            share_of_hate_comments_by_user.append(self.share_of_hate_comments_by_user(comment['uid']))
            print(index)

        print(df['cid'].shape)
        # print(time_since_last_comment_by_user.shape)
        # print(time_since_last_hate_comment_by_user.shape)
        # print(number_of_comments_by_user.shape)
        # print(number_of_hate_comments_by_user.shape)
        # print(share_of_hate_comments_by_user.shape)

        features = np.vstack(
            (df['cid'], time_since_last_comment_by_user, time_since_last_hate_comment_by_user, number_of_comments_by_user, number_of_hate_comments_by_user, share_of_hate_comments_by_user,)).T

        return features

    def __init__(self):
        try:
            self.conn = psycopg2.connect("dbname='hatespeech' user='postgres' host='docker.for.mac.localhost' password='admin'")
        except:
            print('Could not connect to database.')
            sys.exit(0)


    def time_since_last_comment_by_user(self, uid):
        cur = self.conn.cursor()

        query = """ SELECT MAX(created) FROM comments WHERE uid = %s AND hate = 't' """
        cur.execute(query, [uid])

        result = cur.fetchone()

        if result is None or result[0] is None:
            print('Could not find comments for user with uid: ' + str(uid))
            return 0

        return result[0]


    def time_since_last_hate_comment_by_user(self, uid):
        cur = self.conn.cursor()

        query = """ SELECT MAX(created) " +
                "FROM comments " +
                "WHERE uid = %s """
        cur.execute(query, [uid])

        result = cur.fetchone()

        if result is None or result[0] is None:
            print('Could not find hate comments for user with uid: ' + str(uid))
            return 0

        return result[0]


    def number_of_comments_by_user(self, uid):
        cur = self.conn.cursor()

        query = """ SELECT MAX(created) " +
                "FROM comments " +
                "WHERE uid = %s """
        cur.execute(query, [uid])

        result = cur.fetchone()

        if result is None or result[0] is None:
            print('Could not find comments for user with uid: ' + str(uid))
            return 0

        return result[0]


    def number_of_hate_comments_by_user(self, uid):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(cid) " +
                "FROM comments " +
                "WHERE uid = %s AND hate = 't' """
        cur.execute(query, [uid])

        result = cur.fetchone()

        if result is None or result[0] is None:
            print('Could not find hate comments for user with uid: ' + str(uid))
            return 0

        return result[0]


    def share_of_hate_comments_by_user(self, uid):
        if self.number_of_comments_by_user(uid) != 0:
            return self.number_of_hate_comments_by_user(uid) / self.number_of_comments_by_user(uid)
        return self.number_of_hate_comments_by_user(uid)

    def number_of_comments_by_user_on_ressort(self, uid, ressort):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(comments.cid) " +
                "FROM comments, articles " +
                "WHERE comments.uid = %s AND articles.ressort = %s AND comments.url = articles.url """
        cur.execute(query, [uid, ressort])

        result = cur.fetchone()

        if result is None or result[0] is None:
            print('Could not find comments for user with uid: ' + str(uid) + ' on ressort: ' + str(ressort) )
            return 0

        return result[0]


    def number_of_hate_comments_by_user_on_ressort(self, uid, ressort):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(comments.cid) " +
                "FROM comments, articles " +
                "WHERE comments.uid = %s AND hate = 't' AND articles.ressort = %s AND comments.url = articles.url """
        cur.execute(query, [uid, ressort])

        result = cur.fetchone()

        if result is None or result[0] is None:
            print('Could not find hate comments for user with uid: ' + str(uid) + ' on ressort: ' + str(ressort) )
            return 0

        return result[0]


    def share_of_hate_comments_by_user_on_ressort(self, uid, ressort):
        return self.number_of_hate_comments_by_user_on_ressort(uid, ressort) / self.number_of_comments_by_user_on_ressort(uid, ressort)


    def number_of_comments_by_user_since_time(self, uid, time):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(cid) " +
                "FROM comments " +
                "WHERE uid = %s AND created > %s """
        cur.execute(query, [uid, time])

        result = cur.fetchone()

        if result is None or result[0] is None:
            print('Could not find comments for user with uid: ' + str(uid) + ' since: ' + str(time) )
            return 0

        return result[0]


    def number_of_hate_comments_by_user_since_time(self, uid, time):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(cid) " +
                "FROM comments " +
                "WHERE uid = %s AND hate = 't' AND created > %s """
        cur.execute(query, [uid, time])

        result = cur.fetchone()

        if result is None or result[0] is None:
            print('Could not find hate comments for user with uid: ' + str(uid) + ' since: ' + str(time) )
            return 0

        return result[0]


    def share_of_hate_comments_by_user_since_time(self, uid, time):
        return self.number_of_hate_comments_by_user_since_time(uid, time) / self.number_of_comments_by_user_since_time(uid, time)

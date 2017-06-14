from __future__ import division
import psycopg2
import sys
import numpy as np


class UserFeatures:
    def __init__(self):
        try:
            self.conn = psycopg2.connect("dbname='hatespeech' user='postgres' host='localhost' password='admin'")
        except:
            print('Could not connect to database.')
            sys.exit(0)


    def get_number_of_comments_by_uid(self, uid):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(cid) " +
                "FROM comments " +
                "WHERE uid = %s """
        cur.execute(query, [uid])

        result = cur.fetchone()

        if result is None:
            print('Could not find comments for user with uid: ' + str(uid))
            return []

        return result[0]


    def get_number_of_hate_comments_by_uid(self, uid):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(cid) " +
                "FROM comments " +
                "WHERE uid = %s AND hate = 't' """
        cur.execute(query, [uid])

        result = cur.fetchone()

        if result is None:
            print('Could not find hate comments for user with uid: ' + str(uid))
            return []

        return result[0]


    def get_share_of_hate_comments_by_uid(self, uid):
        return get_number_of_hate_comments_by_uid(self, uid) / get_number_of_comments_by_uid(self, uid)


    def get_number_of_comments_by_uid_on_ressort(self, uid, ressort):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(comments.cid) " +
                "FROM comments, articles " +
                "WHERE comments.uid = %s AND articles.ressort = %s AND comments.url = articles.url """
        cur.execute(query, [uid, ressort])

        result = cur.fetchone()

        if result is None:
            print('Could not find comments for user with uid: ' + str(uid) + ' on ressort: ' + str(ressort) )
            return []

        return result[0]


    def get_number_of_hate_comments_by_uid_on_ressort(self, uid, ressort):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(comments.cid) " +
                "FROM comments, articles " +
                "WHERE comments.uid = %s AND hate = 't' AND articles.ressort = %s AND comments.url = articles.url """
        cur.execute(query, [uid, ressort])

        result = cur.fetchone()

        if result is None:
            print('Could not find hate comments for user with uid: ' + str(uid) + ' on ressort: ' + str(ressort) )
            return []

        return result[0]


    def get_share_of_hate_comments_by_uid_on_ressort(self, uid, ressort):
        return get_number_of_hate_comments_by_uid_on_ressort(self, uid, ressort) / get_number_of_comments_by_uid_on_ressort(self, uid, ressort)


    def get_number_of_comments_by_uid_since_time(self, uid, time):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(cid) " +
                "FROM comments " +
                "WHERE uid = %s AND created > %s """
        cur.execute(query, [uid, time])

        result = cur.fetchone()

        if result is None:
            print('Could not find comments for user with uid: ' + str(uid) + ' since: ' + str(time) )
            return []

        return result[0]


    def get_number_of_hate_comments_by_uid_since_time(self, uid, time):
        cur = self.conn.cursor()

        query = """ SELECT COUNT(cid) " +
                "FROM comments " +
                "WHERE uid = %s AND hate = 't' AND created > %s """
        cur.execute(query, [uid, time])

        result = cur.fetchone()

        if result is None:
            print('Could not find hate comments for user with uid: ' + str(uid) + ' since: ' + str(time) )
            return []

        return result[0]


    def get_share_of_hate_comments_by_uid_since_time(self, uid, time):
        return get_number_of_hate_comments_by_uid_since_time(self, uid, time) / get_number_of_comments_by_uid_since_time(self, uid, time)

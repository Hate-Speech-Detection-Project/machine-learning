from __future__ import division
import psycopg2
import sys
import numpy as np


class UserFeatureGenerator:
    def calculate_features_with_dataframe(self, df):
        features = np.vstack((
          df['time_since_last_comment'],
          df['time_since_last_comment_same_user'],
          df['number_of_comments_by_user']
        )).T
        return features

    def __init__(self):
        pass

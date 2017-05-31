from bag_of_words import BagOfWordsClassifier
from text_features import TextFeatureClassifier
from random_forest import RandomForestBOWClassifier
from ada_boost import AdaBoost
import pandas as pd
import numpy as np
from flask import *

class Predictor:
  def initialize(self):
    self.train_df = pd.read_csv('../../data/datasets/1000/train.csv', sep=',')
    self.test_df = pd.read_csv('../../data/datasets/1000/test.csv', sep=',')
    self.bag_of_words_classifier = BagOfWordsClassifier()
    self.bag_of_words_classifier.fit(self.train_df)
    self.text_features_classifier = TextFeatureClassifier()
    self.text_features_classifier.fit(self.train_df)
    self.random_forest_classifier = RandomForestBOWClassifier()
    self.random_forest_classifier.fit(self.train_df)
    self.ada_boost_classifier = AdaBoost()
    self.ada_boost_classifier.fit(self.train_df)

  def accuracy(self):
    self.bow_result = self.bag_of_words_classifier.test(self.test_df)
    self.tf_result = self.text_features_classifier.test(self.test_df)
    self.rf_result = self.random_forest_classifier.test(self.test_df)
    self.ab_result = self.ada_boost_classifier.test(self.test_df)

    bow_result_train = self.bag_of_words_classifier.test(self.test_df)
    tf_result_train = self.text_features_classifier.test(self.test_df)
    rf_result_train = self.random_forest_classifier.test(self.test_df)
    ab_result_train = self.ada_boost_classifier.test(self.test_df)

    bow_accuracy = self.bow_result[0]
    tf_accuracy = self.tf_result[0]
    rf_accuracy = self.rf_result[0]
    ab_accuracy = self.ab_result[0]

    self.ensemble = AdaBoost()
    # ensemble_training_data = pd.DataFrame(data= np.c_[bow_result, tf_result, rf_result, ab_result, self.test_df['hate']]],
    #                  columns= ['bow', 'tf', 'rf', 'ab', 'hate'])
    ensemble_training_data = [bow_result_train[1], tf_result_train[1], rf_result_train[1], ab_result_train[1]]
    print(ensemble_training_data)
    self.ensemble.fitFormatted(ensemble_training_data, self.train_df['hate'])
    ensemble_accuracy = self.ensemble.test(self.test_df)

    return {
      'bag_of_words': np.asscalar(bow_accuracy),
      'text_features': np.asscalar(tf_accuracy),
      'random_forest': np.asscalar(rf_accuracy),
      'ada_boost': np.asscalar(ab_accuracy),
      'ensemble': np.asscalar(ensemble_accuracy)
    }

  def predict(self, comment):
    bow = self.bag_of_words_classifier.predict_with_info(comment)
    tf = self.text_features_classifier.predict(comment)
    rf = self.random_forest_classifier.predict(comment)
    ab = self.ada_boost_classifier.predict(comment)
    ensemble = self.ensemble.predict(comment)

    return {
      'comment': comment,
      'bag_of_words': bow["predicted"][0],
      'hate_words': bow["hate_words"],
      'text_features': tf.tolist(),
      'random_forest': rf,
      'ada_boost': ab,
      'ensemble': ensemble
    }

predictor = Predictor()
print("Learning models...")
predictor.initialize()
print("Done learning models...")

app = Flask(__name__)

@app.route('/')
def hello():
  acc = predictor.accuracy()
  data = {
    'bag_of_words': acc['bag_of_words'],
    'text_features': acc['text_features'],
    'random_forest': acc['random_forest'],
    'ada_boost': acc['ada_boost'],
    'ensemble': acc['ensemble']
  }
  return jsonify(data)

@app.route('/predict', methods=["POST", "GET"])
def predict():
  if request.method == "POST":
    json_dict = request.get_json()
    comment = json_dict['comment']
  else:
    comment = request.args.get('comment', '')

  result = predictor.predict(comment)
  return jsonify(result)

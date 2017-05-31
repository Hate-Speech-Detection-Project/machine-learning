from bag_of_words import BagOfWordsClassifier
from text_features import TextFeatureClassifier
import pandas as pd
import numpy as np
from flask import *

class Predictor:
  def initialize(self):
    # self.train_df = pd.read_csv('../../data/test.csv', sep=',')
    # self.test_df = pd.read_csv('../../data/tiny.csv', sep=',')
    self.train_df = pd.read_csv('../../data/datasets/stratified/train.csv', sep=',')
    self.test_df = pd.read_csv('../../data/datasets/stratified/test.csv', sep=',')
    self.bag_of_words_classifier = BagOfWordsClassifier()
    self.bag_of_words_classifier.fit(self.train_df)
    self.text_features_classifier = TextFeatureClassifier()
    self.text_features_classifier.fit(self.train_df)

  def accuracy(self):
    bow_accuracy = self.bag_of_words_classifier.test(self.test_df)
    tf_accuracy = self.text_features_classifier.test(self.test_df)
    return {
      'bag_of_words': np.asscalar(bow_accuracy),
      'text_features': np.asscalar(tf_accuracy)
    }

  def predict(self, comment):
    bow = self.bag_of_words_classifier.predict_with_info(comment)
    tf = self.text_features_classifier.predict(comment)
    
    return {
      'comment': comment,
      'bag_of_words': bow["predicted"][0],
      'hate_words': bow["hate_words"],
      'text_features': tf.tolist()
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
    'text_features': acc['text_features']
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

# app.run(host='0.0.0.0', port=9999)
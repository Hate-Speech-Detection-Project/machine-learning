from bag_of_words import BagOfWordsClassifier
from text_features import TextFeatureClassifier
from random_forest import RandomForestBOWClassifier
from vote import Vote
from ada_boost import AdaBoost
from preprocessor import PreProcessor
import pandas as pd
import numpy as np
from threading import Thread
from flask import *

class Predictor:
  def initialize(self):
    self.threads = []
    self.preprocessor = PreProcessor()

    self.train_df = pd.read_csv('../../data/datasets/stratified_dual_small/train.csv', sep=',')
    self.test_df = pd.read_csv('../../data/datasets/stratified_dual_small/test1.csv', sep=',')
    self.test_ensemble_df = pd.read_csv('../../data/datasets/stratified_dual_small/test1.csv', sep=',')
    # self.train_df = pd.read_csv('../../data/datasets/1000/train.csv', sep=',')
    # self.test_df = pd.read_csv('../../data/datasets/1000/test.csv', sep=',')
    # self.test_ensemble_df = pd.read_csv('../../data/datasets/10000/test.csv', sep=',')

    bag_of_words_features_array = self.preprocessor.trainFeatureMatrix(self.train_df);

    print(bag_of_words_features_array.shape)

    self.bag_of_words_classifier = BagOfWordsClassifier()
    thread = Thread(target = self.bag_of_words_classifier.fit, args = (self.train_df,))
    self.threads.append(thread)
    thread.start()

    self.text_features_classifier = TextFeatureClassifier()
    thread = Thread(target = self.text_features_classifier.fit, args = (self.train_df,))
    self.threads.append(thread)
    thread.start()

    self.random_forest_classifier = RandomForestBOWClassifier(self.preprocessor)
    thread = Thread(target = self.random_forest_classifier.fitFormatted, args = (bag_of_words_features_array, self.train_df['hate']))
    self.threads.append(thread)
    thread.start()

    self.ada_boost_classifier = AdaBoost(self.preprocessor)
    thread = Thread(target = self.ada_boost_classifier.fitFormatted, args = (bag_of_words_features_array, self.train_df['hate']))
    self.threads.append(thread)
    thread.start()

    for thread in self.threads:
        thread.join()

  def fitClassifier(self, classifier, df):
    classifier.fit(df)

  def accuracy(self):
    self.bow_result = self.bag_of_words_classifier.test(self.test_df)
    self.tf_result = self.text_features_classifier.test(self.test_df)
    self.rf_result = self.random_forest_classifier.test(self.test_df)
    self.ab_result = self.ada_boost_classifier.test(self.test_df)

    # these names are totally misleading... change them
    bow_result_train = self.bag_of_words_classifier.test(self.test_ensemble_df)
    tf_result_train = self.text_features_classifier.test(self.test_ensemble_df)
    rf_result_train = self.random_forest_classifier.test(self.test_ensemble_df)
    ab_result_train = self.ada_boost_classifier.test(self.test_ensemble_df)

    # use testresults that also participate in the ensemble
    bow_accuracy = bow_result_train[0]
    tf_accuracy = tf_result_train[0]
    rf_accuracy = rf_result_train[0]
    ab_accuracy = ab_result_train[0]

    self.ensemble = AdaBoost(PreProcessor())

    ensemble_training_data = np.matrix((self.preprocessor.convertBoolStringsToNumbers(self.bow_result[1]),
                              self.preprocessor.convertBoolStringsToNumbers(self.tf_result[1]),
                              self.preprocessor.convertBoolStringsToNumbers(self.rf_result[1]),
                              self.preprocessor.convertBoolStringsToNumbers(self.ab_result[1]))).getT()

    ensemble_test_data = np.matrix((self.preprocessor.convertBoolStringsToNumbers(bow_result_train[1]),
                          self.preprocessor.convertBoolStringsToNumbers(tf_result_train[1]),
                          self.preprocessor.convertBoolStringsToNumbers(rf_result_train[1]),
                          self.preprocessor.convertBoolStringsToNumbers(ab_result_train[1]))).getT()

    print(ensemble_training_data)
    self.ensemble.fitFormatted(ensemble_training_data, self.test_df['hate'])

    ensemble_results = self.ensemble.testFeatuerMatrix(ensemble_test_data, self.test_ensemble_df['hate'])

    ensemble_analysis = np.array([self.preprocessor.convertBoolStringsToNumbers(bow_result_train[1]),
                                self.preprocessor.convertBoolStringsToNumbers(tf_result_train[1]),
                                self.preprocessor.convertBoolStringsToNumbers(rf_result_train[1]),
                                self.preprocessor.convertBoolStringsToNumbers(ab_result_train[1]),
                                self.preprocessor.convertBoolStringsToNumbers(ensemble_results[1]),
                                self.preprocessor.convertBoolStringsToNumbers(self.test_ensemble_df['hate'])]).T

    df = pd.DataFrame(data=ensemble_analysis[0:,0:],    # values
                        index=self.test_ensemble_df['cid'],    # 1st column as index
                        columns=['bow', 'tf', 'rf', 'ab', 'ensemble', 'hate'])
    df.to_csv('ensemble_analysis.csv', sep=';', encoding='utf-8')

    #### Voter

    voter = Vote(1)
    voter.fitFormatted(ensemble_test_data)
    voter_results = voter.getResults(self.preprocessor.convertBoolStringsToNumbers(self.test_ensemble_df['hate']))

    print(voter_results)

    return {
      'bag_of_words': np.asscalar(bow_accuracy),
      'text_features': np.asscalar(tf_accuracy),
      'random_forest': np.asscalar(rf_accuracy),
      'ada_boost': np.asscalar(ab_accuracy),
      'ensemble': np.asscalar(ensemble_results[0]),
      'voter': voter_results[0]
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
    'ensemble': acc['ensemble'],
    'voter': acc['voter']
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

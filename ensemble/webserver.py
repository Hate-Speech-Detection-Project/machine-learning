import matplotlib.pyplot as plt
from bag_of_words import BagOfWordsClassifier
from text_features import TextFeatureClassifier
from random_forest import RandomForestBOWClassifier
from vote import Vote
from ada_boost import AdaBoost
from preprocessor import Preprocessor
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from threading import Thread
from flask import *
from utils import CorrelationMatrix
import json
import io
import base64

class Predictor:
  def initialize(self):
    self.threads = []
    self.preprocessor = Preprocessor()

    self.bow_result = None
    self.tf_result = None
    self.rf_result = None
    self.ab_result = None

    #self.train_df = pd.read_csv('../../data/datasets/stratified_dual/train.csv', sep=',')
    #self.test_df = pd.read_csv('../../data/datasets/stratified_dual/test1.csv', sep=',')
    #self.test_ensemble_df = pd.read_csv('../../data/datasets/stratified_dual/test1.csv', sep=',')
    self.train_df = pd.read_csv('../../data/datasets/stratified_dual_small/train.csv', sep=',')
    self.test_df = pd.read_csv('../../data/datasets/stratified_dual_small/test1.csv', sep=',')
    self.test_ensemble_df = pd.read_csv('../../data/datasets/stratified_dual_small/test1.csv', sep=',')


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
    thread = Thread(target = self.random_forest_classifier.fitFeatureMatrix, args = (bag_of_words_features_array, self.train_df['hate']))
    self.threads.append(thread)
    thread.start()

    self.ada_boost_classifier = AdaBoost(self.preprocessor)
    thread = Thread(target = self.ada_boost_classifier.fitFeatureMatrix, args = (bag_of_words_features_array, self.train_df['hate']))
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

    bow_ensemble_test = self.bag_of_words_classifier.test(self.test_ensemble_df)
    tf_ensemble_test = self.text_features_classifier.test(self.test_ensemble_df)
    rf_ensemble_test = self.random_forest_classifier.test(self.test_ensemble_df)
    ab_ensemble_test = self.ada_boost_classifier.test(self.test_ensemble_df)

    # use testresults that also participate in the ensemble
    bow_accuracy = bow_ensemble_test[0]
    tf_accuracy = tf_ensemble_test[0]
    rf_accuracy = rf_ensemble_test[0]
    ab_accuracy = ab_ensemble_test[0]

    self.ensemble = AdaBoost(Preprocessor())

    ensemble_training_data = np.matrix((self.bow_result[2],
                                        self.tf_result[1],
                                        self.rf_result[2],
                                        self.ab_result[2])).getT()

    ensemble_test_data = np.matrix((bow_ensemble_test[2],
                          tf_ensemble_test[1],
                          rf_ensemble_test[2],
                          ab_ensemble_test[2])).getT()

    print(ensemble_training_data)
    self.ensemble.fitFeatureMatrix(ensemble_training_data, self.test_df['hate'])

    ensemble_results = self.ensemble.testFeatuerMatrix(ensemble_test_data, self.test_ensemble_df['hate'])

    ensemble_analysis = np.array([Preprocessor.convertBoolStringsToNumbers(bow_ensemble_test[1]),
                                Preprocessor.convertBoolStringsToNumbers(tf_ensemble_test[1]),
                                Preprocessor.convertBoolStringsToNumbers(rf_ensemble_test[1]),
                                Preprocessor.convertBoolStringsToNumbers(ab_ensemble_test[1]),
                                Preprocessor.convertBoolStringsToNumbers(ensemble_results[1]),
                                Preprocessor.convertBoolStringsToNumbers(self.test_ensemble_df['hate'])]).T

    df = pd.DataFrame(data=ensemble_analysis[0:,0:],    # values
                        index=self.test_ensemble_df['cid'],    # 1st column as index
                        columns=['bow', 'tf', 'rf', 'ab', 'ensemble', 'hate'])
    df.to_csv('ensemble_analysis.csv', sep=';', encoding='utf-8')

    #### Voter

    voter_1 = Vote(1)
    voter_1.fitFeatureMatrix(ensemble_test_data)
    voter_1_results = voter_1.getResults(Preprocessor.convertBoolStringsToNumbers(self.test_ensemble_df['hate']))

    voter_2 = Vote(2)
    voter_2.fitFeatureMatrix(ensemble_test_data)
    voter_2_results = voter_2.getResults(Preprocessor.convertBoolStringsToNumbers(self.test_ensemble_df['hate']))

    voter_3 = Vote(3)
    voter_3.fitFeatureMatrix(ensemble_test_data)
    voter_3_results = voter_3.getResults(Preprocessor.convertBoolStringsToNumbers(self.test_ensemble_df['hate']))

    voter_4 = Vote(4)
    voter_4.fitFeatureMatrix(ensemble_test_data)
    voter_4_results = voter_4.getResults(Preprocessor.convertBoolStringsToNumbers(self.test_ensemble_df['hate']))

    return {
      'bag_of_words': bow_ensemble_test[0].toString(),
      'text_features': tf_ensemble_test[0].toString(),
      'random_forest': rf_ensemble_test[0].toString(),
      'ada_boost': ab_ensemble_test[0].toString(),
      'ensemble': ensemble_results[0].toString(),
      'voter(1)': voter_1_results[0].toString(),
      'voter(2)': voter_2_results[0].toString(),
      'voter(3)': voter_3_results[0].toString(),
      'voter(4)': voter_4_results[0].toString()
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

angle = 0

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
    'voter(1)': acc['voter(1)'],
    'voter(2)': acc['voter(2)'],
    'voter(3)': acc['voter(3)'],
    'voter(4)': acc['voter(4)']
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

@app.route('/correlation')
def correlation():
  dataRows = [predictor.bow_result[2],
              predictor.rf_result[2],
              predictor.ab_result[2],
              predictor.bow_result[2]]
  correlationMatrix = CorrelationMatrix(dataRows)
  return jsonify(correlationMatrix.get())


@app.route('/plot')
def plot():

    img = io.BytesIO()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    for index,observation in enumerate(predictor.bow_result[2]):
      ax.scatter( predictor.rf_result[2][index], 
                  predictor.ab_result[2][index], 
                  predictor.bow_result[2][index], 
                  alpha=np.mean((predictor.rf_result[2][index], predictor.ab_result[2][index], predictor.bow_result[2][index]))/2,
                  s=30)
      ax.scatter( predictor.rf_result[2][index], 
                  predictor.ab_result[2][index], 
                  predictor.bow_result[2][index], 
                  s = 1000,
                  alpha=np.mean((predictor.rf_result[2][index], predictor.ab_result[2][index], predictor.bow_result[2][index])),
                  marker=r"$ {} $".format(predictor.test_df['cid'][index]))

    

    ax.set_xlabel('Random Forest')
    ax.set_ylabel('Ada Boost')
    ax.set_zlabel('Naive Bayes')

    # ax.view_init(30, angle)
    # angle += 10

    # plt.plot(predictor.rf_result[2], predictor.ab_result[2], 'ro')
    plt.savefig(img, format='png')

    img.seek(0)

    return send_file(img, mimetype='image/png')



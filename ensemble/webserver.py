from bag_of_words import BagOfWordsClassifier
from text_features import TextFeatureClassifier
import pandas as pd
import numpy as np
from flask import *
from crawler.article_crawler import ArticelCrawler
from article_features import ArticleFeatures



class Predictor:
    def initialize(self):

        # self.train_df = pd.read_csv('../../data/test.csv', sep=',')
        # self.test_df = pd.read_csv('../../data/tiny.csv', sep=',')
        # self.train_df = pd.read_csv('../data/stratified/train.csv', sep=',')
        self.train_df = pd.read_csv('../data/1000/train.csv', sep=',')
        self.test_df = pd.read_csv('../data/stratified/test.csv', sep=',')

        self.bag_of_words_classifier = BagOfWordsClassifier()
        self.bag_of_words_classifier.fit(self.train_df)
        self.text_features_classifier = TextFeatureClassifier()
        self.text_features_classifier.fit(self.train_df)

        self.article_features = ArticleFeatures()
        ArticelCrawler.create_crawler()

    def accuracy(self):
        bow_accuracy = self.bag_of_words_classifier.test(self.test_df)
        tf_accuracy = self.text_features_classifier.test(self.test_df)
        return {
            'bag_of_words': np.asscalar(bow_accuracy),
            'text_features': np.asscalar(tf_accuracy)
        }

    def predict(self, comment,timestamp):
        bow = self.bag_of_words_classifier.predict_with_info(comment)
        tf = self.text_features_classifier.predict(comment,timestamp)

        return {
            'comment': comment,
            'bag_of_words': bow["predicted"][0],
            'hate_words': bow["hate_words"],
            'text_features': tf.tolist()
        }

    def clean_comment(self,comment,url):
        ArticelCrawler.start_crawler(url)
        article = ArticelCrawler.crawled_article
        if article is None or article.get_body() is None:
            print("Could not get article for comment. Continues without a change.")
            return comment

        comment = self.article_features.remove_words_of_comment_by_given_text(comment, article.get_body())
        return comment



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

    timestamp = 0.0
    if request.method == "POST":
        json_dict = request.get_json()

        comment = json_dict['comment']

        if 'url' in json_dict:
            comment = predictor.clean_comment(comment,json_dict['url'])

        if 'created' in json_dict:
            timestamp = int(json_dict['created'])

    else:
        comment = request.args.get('comment', '')

    result = predictor.predict(comment, timestamp)
    return jsonify(result)


app.run(host='localhost', port=9999)

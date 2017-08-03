#in case you get problems with some qt4 / qt5 pyqtobject stuff
#import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from flask import *
from flask_cors import CORS, cross_origin
from utils import AnalysisInformation, CorrelationMatrix
from ensembleclassifier import EnsembleClassifier
import json
import io
import base64

# trainDf = pd.read_csv('../data/stratified_dual_smallest/test1.csv', sep=',')
# testDf = pd.read_csv('../data/stratified_dual_smallest/test1.csv', sep=',')
# testEnsembleDf = pd.read_csv('../data/stratified_dual_smallest/test2.csv', sep=',')

trainDf = pd.read_csv('../data/1000/train.csv', sep=',')
testDf = pd.read_csv('../data/1000/test.csv', sep=',')
testEnsembleDf = pd.read_csv('../data/1000/test.csv', sep=',')

predictor = EnsembleClassifier()
predictor.initClassifiers(trainDf, testDf, testEnsembleDf, 'hate')
print('Learning models...')
predictor.fitClassifiers()
print('Done learning models...')
print('Testing models...')
predictor.testClassifiers()
print('Done Testing models...')
predictor.initEnsembleClassifier()

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():

  data = {}

  for featureSetName in predictor.getFeatureSetNames():
    featureSet = {}

    for classifierName in predictor.getClassifierNames():
      featureSet[classifierName] = predictor.getClassifierStatistics(featureSetName, classifierName)[0].toString()

    data[featureSetName] = featureSet

  return jsonify(data)

@app.route('/correlation')
def correlation():
  dataRows = {}

  for featureSetName in predictor.getFeatureSetNames():
    for classifierName in predictor.getClassifierNames():
      dataRows[featureSetName[:1] + classifierName] = predictor.getClassifierStatistics(featureSetName, classifierName)[2]

  correlationMatrix = CorrelationMatrix(dataRows)
  return correlationMatrix.toString()

@app.route('/single/<cid>')
def single(cid):
	return testDf[testDf['cid'] == int(cid)]

@app.route('/predict', methods=["POST", "GET"])
def predict():
  if request.method == "POST":
    json_dict = request.get_json()
    print(json_dict)
    comment = json_dict['comment']
    url = json_dict['url']
  else:
    comment = request.args.get('comment', '')
    url = request.args.get('url', '')
  results = predictor.testClassifiersSingle(comment, url)
  print(results)
  return jsonify(results)

@app.route('/plot')
def plot():

    img = io.BytesIO()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    for index,observation in enumerate(predictor.getClassifierStatistics('BOW Ensemble Test', 'AdaBoost')[2]):
      ax.scatter( predictor.getClassifierStatistics('BOW Ensemble Test', 'AdaBoost')[2][index], 
                  predictor.getClassifierStatistics('TextFeatures Ensemble Test', 'AdaBoost')[2][index], 
                  predictor.getClassifierStatistics('UserFeatures Ensemble Test', 'AdaBoost')[2][index], 
                  alpha=np.mean((predictor.getClassifierStatistics('BOW Ensemble Test','AdaBoost')[2][index], predictor.getClassifierStatistics('TextFeatures Ensemble Test', 'AdaBoost')[2][index], predictor.getClassifierStatistics('UserFeatures Ensemble Test', 'AdaBoost')[2][index]))/2,
                  s=30)
      ax.scatter( predictor.getClassifierStatistics('BOW Ensemble Test', 'AdaBoost')[2][index], 
                  predictor.getClassifierStatistics('TextFeatures Ensemble Test', 'AdaBoost')[2][index], 
                  predictor.getClassifierStatistics('UserFeatures Ensemble Test', 'AdaBoost')[2][index], 
                  s = 1000,
                  alpha=np.mean((predictor.getClassifierStatistics('BOW Ensemble Test', 'AdaBoost')[2][index], predictor.getClassifierStatistics('TextFeatures Ensemble Test', 'AdaBoost')[2][index], predictor.getClassifierStatistics('UserFeatures Ensemble Test', 'AdaBoost')[2][index])),
                  marker=r'$ {} $'.format(testDf['cid'][index]))

    

    ax.set_xlabel('BOW Ensemble')
    ax.set_ylabel('TextFeatures Ensemble')
    ax.set_zlabel('UserFeatures Ensemble Bayes')

    # ax.view_init(30, angle)
    # angle += 10

    # plt.plot(predictor.rf_result[2], predictor.ab_result[2], 'ro')
    plt.savefig(img, format='png')

    img.seek(0)

    return send_file(img, mimetype='image/png')


# return the UI
@app.route('/index')
def index():
    return render_template('index.html')

app.run(host='0.0.0.0')

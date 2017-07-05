import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from flask import *
from utils import AnalysisInformation
from ensembleclassifier import EnsembleClassifier
import json
import io
import base64

trainDf = pd.read_csv('../../data/datasets/stratified_dual_smallest/train.csv', sep=',')
testDf = pd.read_csv('../../data/datasets/stratified_dual_smallest/test1.csv', sep=',')
testEnsembleDf = pd.read_csv('../../data/datasets/stratified_dual_smallest/test2.csv', sep=',')


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

@app.route('/')
def hello():

	print(predictor.getClassifierStatistics('BOW', 'RandomForest')[0])

	data = {
		'BOW' : {
			'Random Forest': predictor.getClassifierStatistics('BOW', 'RandomForest')[0].toString(),
			'Ada Boost': predictor.getClassifierStatistics('BOW', 'AdaBoost')[0].toString(),
			'Naive Bayes': predictor.getClassifierStatistics('BOW', 'Naive Bayes')[0].toString()
		},
		'Textfeatures' : {
			'Random Forest': predictor.getClassifierStatistics('TextFeatures', 'RandomForest')[0].toString(),
			'Ada Boost': predictor.getClassifierStatistics('TextFeatures', 'AdaBoost')[0].toString(),
			'Naive Bayes': predictor.getClassifierStatistics('TextFeatures', 'Naive Bayes')[0].toString()
		},
		'Ensemble' : {
			'Random Forest': predictor.getClassifierStatistics('Ensemble', 'RandomForest')[0].toString(),
			'Ada Boost': predictor.getClassifierStatistics('Ensemble', 'AdaBoost')[0].toString(),
			'Naive Bayes': predictor.getClassifierStatistics('Ensemble', 'Naive Bayes')[0].toString()
		}
	}
	return jsonify(data)

@app.route('/correlation')
def correlation():
  dataRows = [predictor.getClassifierStatistics('BOW', 'RandomForest')[2],
              predictor.getClassifierStatistics('BOW', 'AdaBoost')[2],
              predictor.getClassifierStatistics('BOW', 'Naive Bayes')[2]]
  correlationMatrix = CorrelationMatrix(dataRows)
  return jsonify(correlationMatrix.get())

@app.route('/plot')
def plot():

    img = io.BytesIO()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    for index,observation in enumerate(predictor.getClassifierStatistics('BOW', 'RandomForest')[2]):
      ax.scatter( predictor.getClassifierStatistics('BOW', 'AdaBoost')[2][index], 
                  predictor.getClassifierStatistics('BOW', 'Naive Bayes')[2][index], 
                  predictor.getClassifierStatistics('BOW', 'RandomForest')[2][index], 
                  alpha=np.mean((predictor.getClassifierStatistics('BOW','AdaBoost')[2][index], predictor.getClassifierStatistics('BOW', 'Naive Bayes')[2][index], predictor.getClassifierStatistics('BOW', 'RandomForest')[2][index]))/2,
                  s=30)
      ax.scatter( predictor.getClassifierStatistics('BOW', 'AdaBoost')[2][index], 
                  predictor.getClassifierStatistics('BOW', 'Naive Bayes')[2][index], 
                  predictor.getClassifierStatistics('BOW', 'RandomForest')[2][index], 
                  s = 1000,
                  alpha=np.mean((predictor.getClassifierStatistics('BOW', 'AdaBoost')[2][index], predictor.getClassifierStatistics('BOW', 'Naive Bayes')[2][index], predictor.getClassifierStatistics('BOW', 'RandomForest')[2][index])),
                  marker=r'$ {} $'.format(testDf['cid'][index]))

    

    ax.set_xlabel('Random Forest')
    ax.set_ylabel('Ada Boost')
    ax.set_zlabel('Naive Bayes')

    # ax.view_init(30, angle)
    # angle += 10

    # plt.plot(predictor.rf_result[2], predictor.ab_result[2], 'ro')
    plt.savefig(img, format='png')

    img.seek(0)

    return send_file(img, mimetype='image/png')
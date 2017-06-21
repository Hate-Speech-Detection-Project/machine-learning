import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from flask import *
from utils import CorrelationMatrix
from ensembleclassifier import EnsembleClassifier
import json
import io
import base64

trainDf = pd.read_csv('../../data/datasets/stratified_dual_small/train.csv', sep=',')
testDf = pd.read_csv('../../data/datasets/stratified_dual_small/test1.csv', sep=',')
testEnsembleDf = pd.read_csv('../../data/datasets/stratified_dual_small/test1.csv', sep=',')
predictor = EnsembleClassifier()
predictor.initClassifiers()
print("Learning models...")
predictor.fitClassifiers(trainDf, trainDf['hate'])
print("Done learning models...")
print("Testing models...")
predictor.testClassifiers(testDf, testDf['hate'])
print("Done Testing models...")

app = Flask(__name__)

@app.route('/')
def hello():
  data = {
    'Random Forest': predictor.getClassifierStatistics("RandomForest")[0].toString(),
    'Ada Boost': predictor.getClassifierStatistics("AdaBoost")[0].toString(),
    'Naive Bayes': predictor.getClassifierStatistics("Naive Bayes")[0].toString()
  }
  return jsonify(data)

@app.route('/correlation')
def correlation():
  dataRows = [predictor.getClassifierStatistics("RandomForest")[2],
              predictor.getClassifierStatistics("AdaBoost")[2],
              predictor.getClassifierStatistics("Naive Bayes")[2]]
  correlationMatrix = CorrelationMatrix(dataRows)
  return jsonify(correlationMatrix.get())

@app.route('/plot')
def plot():

    img = io.BytesIO()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    for index,observation in enumerate(predictor.getClassifierStatistics("RandomForest")[2]):
      ax.scatter( predictor.getClassifierStatistics("AdaBoost")[2][index], 
                  predictor.getClassifierStatistics("Naive Bayes")[2][index], 
                  predictor.getClassifierStatistics("RandomForest")[2][index], 
                  alpha=np.mean((predictor.getClassifierStatistics("AdaBoost")[2][index], predictor.getClassifierStatistics("Naive Bayes")[2][index], predictor.getClassifierStatistics("RandomForest")[2][index]))/2,
                  s=30)
      ax.scatter( predictor.getClassifierStatistics("AdaBoost")[2][index], 
                  predictor.getClassifierStatistics("Naive Bayes")[2][index], 
                  predictor.getClassifierStatistics("RandomForest")[2][index], 
                  s = 1000,
                  alpha=np.mean((predictor.getClassifierStatistics("AdaBoost")[2][index], predictor.getClassifierStatistics("Naive Bayes")[2][index], predictor.getClassifierStatistics("RandomForest")[2][index])),
                  marker=r"$ {} $".format(testDf['cid'][index]))

    

    ax.set_xlabel('Random Forest')
    ax.set_ylabel('Ada Boost')
    ax.set_zlabel('Naive Bayes')

    # ax.view_init(30, angle)
    # angle += 10

    # plt.plot(predictor.rf_result[2], predictor.ab_result[2], 'ro')
    plt.savefig(img, format='png')

    img.seek(0)

    return send_file(img, mimetype='image/png')
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
    'SVM': predictor.getClassifierStatistics("SVM")[0].toString()
  }
  return jsonify(data)

@app.route('/correlation')
def correlation():
  dataRows = [predictor.bow_result[2],
              predictor.rf_result[2],
              predictor.ab_result[2],
              predictor.bow_result[2]]
  correlationMatrix = CorrelationMatrix(dataRows)
  return jasonify(correlationMatrix.get())
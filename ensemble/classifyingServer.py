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

class EnsembleClassifier:
	def initialize(self):
	    self.threads = []
	    self.preprocessor = Preprocessor()
	    self.scheduler = Scheduler()
	    self.classifiers = {}

	    self.trainingFeatureArray = None
	    self.trainingGroundTruth = None

	    self.testFeatureArray = None
	    self.testGroundTruth = None

	def __addClassifier(self, name, classifier):
		self.classifiers[name] = classifier

	def __fitClassifiers(self, featureArray):
		for classifier in self.classifiers:
			self.scheduler.schedule(classifier.fitFeatureArray, featureArray)
		self.scheduler.joinAll()

	def __testClassifiers(self, featureArray, groundTruth):
		for classifier in self.classifiers:
			self.scheduler.schedule(classifier.test, featureArray)

	def __generateTrainingfeatures(self, trainDf):
		self.trainFeatureMatrix = self.preprocessor.trainFeatureMatrix(trainDf)

	def initClassifiers(self):
		self.__addClassifier("RandomForest", RandomForestBOWClassifier(self.preprocessor))
		self.__addClassifier("AdaBoost", AdaBoost(self.preprocessor))

	def fitClassifiers(trainDf, groundTruth):
		self.scheduler.schedule(self.__generateTrainingfeatures, trainDf)
		self.groundTruth = groundTruth
		self.__fitClassifiers(self.trainingFeatureArray, self.trainingGroundTruth)

	def testClassifiers(trainDf, groundTruth):
		self.scheduler.schedule(self.__generateTrainingfeatures, trainDf)
		self.groundTruth = groundTruth
		self.__fitClassifiers(self.trainingFeatureArray, self.trainingGroundTruth)

	def getClassifierStatistics(self, classifierName):
		return self.classifiers[classifierName].test(None)


trainDf = pd.read_csv('../../data/datasets/stratified_dual_small/train.csv', sep=',')
testDf = pd.read_csv('../../data/datasets/stratified_dual_small/test1.csv', sep=',')
testEnsembleDf = pd.read_csv('../../data/datasets/stratified_dual_small/test1.csv', sep=',')
predictor = EnsembleClassifier()
print("Learning models...")
predictor.initClassifiers()
predictor.fitClassifiers(trainDf, trainDf["hate"])
predictor.testClassifiers(testDf, testDf["hate"])
print("Done learning models...")

angle = 0

app = Flask(__name__)

@app.route('/')
def hello():
  acc = predictor.accuracy()
  data = {
    'bag_of_words': predictor.getClassifierStatistics("RandomForest"),
    'ada_boost': predictor.getClassifierStatistics("AdaBoost"),
  }
  return jsonify(data)
from bag_of_words import BagOfWordsClassifier
from random_forest import RandomForestBOWClassifier
from vote import Vote
from ada_boost import AdaBoost
from preprocessor import Preprocessor
from mpl_toolkits.mplot3d import Axes3D
from scheduler import Scheduler
import pandas as pd
import numpy as np
from utils import CorrelationMatrix

class EnsembleClassifier:
	def __init__(self):
		self.threads = []
		self.preprocessor = Preprocessor()
		self.scheduler = Scheduler()
		self.classifiers = {}

		self.trainingFeatureMatrix = {}
		self.trainingGroundTruth = {}
		self.testFeatureMatrix = {}
		self.testGroundTruth = {}

	def __addClassifier(self, name, classifier):
		self.classifiers[name] = classifier

	def __fitClassifiers(self, featureArray, groundTruth):
		for key, classifier in self.classifiers.items():
			self.scheduler.schedule(function = classifier.fitFeatureMatrix, args = (featureArray, groundTruth))
		self.scheduler.joinAll()

	def __testClassifiers(self, featureArray, groundTruth):
		for key, classifier in self.classifiers.items():
			self.scheduler.schedule(function = classifier.testFeatureMatrix, args = (featureArray, groundTruth))
		self.scheduler.joinAll();		

	def __generateTrainingFeatures(self, trainingDf):
		if not 'BOW' in self.trainingFeatureMatrix.keys():
			self.trainingFeatureMatrix['BOW'] = self.preprocessor.trainFeatureMatrix(trainingDf)

	def __generateTestFeatures(self, testDf):
		if not 'BOW' in self.testFeatureMatrix.keys():
			self.testFeatureMatrix['BOW'] = self.preprocessor.createFeatureMatrix(testDf)

	def initClassifiers(self):
		self.__addClassifier("RandomForest", RandomForestBOWClassifier(self.preprocessor))
		self.__addClassifier("AdaBoost", AdaBoost(self.preprocessor))
		self.__addClassifier("Naive Bayes", BagOfWordsClassifier())

	def fitClassifiers(self, trainDf, groundTruth):
		self.__generateTrainingFeatures(trainDf)
		self.trainingGroundTruth['BOW'] = groundTruth
		self.__fitClassifiers(self.trainingFeatureMatrix['BOW'], self.trainingGroundTruth['BOW'])

	def testClassifiers(self, testDf, groundTruth):
		self.__generateTestFeatures(testDf)
		self.testGroundTruth['BOW'] = groundTruth
		self.__testClassifiers(self.testFeatureMatrix['BOW'], self.testGroundTruth['BOW'])

	def getClassifierStatistics(self, classifierName):
		return self.classifiers[classifierName].testFeatureMatrix(None, None)
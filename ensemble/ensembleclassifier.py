from bag_of_words import BagOfWordsClassifier
from text_features import TextFeatureClassifier
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

		self.trainingFeatureMatrix = None
		self.trainingGroundTruth = None
		self.testFeatureMatrix = None
		self.testGroundTruth = None

	def __addClassifier(self, name, classifier):
		self.classifiers[name] = classifier

	def __fitClassifiers(self, featureArray, groundTruth):
		for key, classifier in self.classifiers.items():
			self.scheduler.schedule(function = classifier.fitFeatureArray, args = (featureArray, groundTruth))
		self.scheduler.joinAll()

	def __testClassifiers(self, featureArray, groundTruth):
		for key, classifier in self.classifiers.items():
			self.scheduler.schedule(function = classifier.testFeatureArray, args = (featureArray, groundTruth))
		self.scheduler.joinAll();		

	def __generateTrainingFeatures(self, trainingDf):
		if self.trainingFeatureMatrix == None:
			self.trainingFeatureMatrix = self.preprocessor.trainFeatureMatrix(trainingDf)

	def __generateTestFeatures(self, testDf):
		if self.testFeatureMatrix == None:
			self.testFeatureMatrix = self.preprocessor.createFeatureMatrix(testDf)

	def initClassifiers(self):
		self.__addClassifier("RandomForest", RandomForestBOWClassifier(self.preprocessor))
		self.__addClassifier("AdaBoost", AdaBoost(self.preprocessor))
		self.__addClassifier("Naive Bayes", BagOfWordsClassifier())

	def fitClassifiers(self, trainDf, groundTruth):
		self.__generateTrainingFeatures(trainDf)
		self.trainingGroundTruth = groundTruth
		self.__fitClassifiers(self.trainingFeatureMatrix, self.trainingGroundTruth)

	def testClassifiers(self, testDf, groundTruth):
		self.__generateTestFeatures(testDf)
		self.testGroundTruth = groundTruth
		self.__testClassifiers(self.testFeatureMatrix, self.testGroundTruth)

	def getClassifierStatistics(self, classifierName):
		return self.classifiers[classifierName].testFeatureArray(None, None)
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
from text_features import TextFeatureGenerator

class EnsembleClassifier:
	def __init__(self):
		self.threads = []
		self.preprocessor = Preprocessor()
		self.textFeatureGenerator = TextFeatureGenerator()

		self.scheduler = Scheduler()
		self.classifiers = {}

		self.trainingDataFrame = None
		self.testDataFrame = None

		self.featureTrainingGen = {}
		self.featureTestGen = {}

		# processed featuresets
		self.trainingFeatureMatrix = {}
		self.trainingGroundTruth = {}

		self.testFeatureMatrix = {}
		self.testGroundTruth = {}

	def __addClassifier(self, name, classifier):
		for featureSet in self.trainingfeatureSets:
			if not featureSet in self.classifiers:
				self.classifiers[featureSet] = {}
			self.classifiers[name].append(key, classifier)

	def __fitClassifiers(self):
		for featureSet in self.trainingfeatureSets:
			for key, classifier in self.classifiers[featureSet].items():
				self.scheduler.schedule(function = classifier.fitFeatureMatrix, 
										args = (self.trainingFeatureMatrix[featureSet], 
												self.trainingGroundTruth[featureSet]))
		self.scheduler.joinAll()

	def __testClassifiers(self):
		for featureSet in self.testfeatureSets:
			for key, classifier in self.classifiers.items():
				self.scheduler.schedule(function = classifier.testFeatureMatrix, 
										args = (self.testFeatureMatrix[featureSet], 
												self.testgroundTruth[featureSet]))
		self.scheduler.joinAll();		

	def __generateTrainingFeatures(self):
		for key, conversion in self.featureTrainingGen.items():
			if not key in self.trainingFeatureMatrix.keys():
				# self.trainingFeatureMatrix[key] = self.preprocessor.trainFeatureMatrix(featureSet)
				self.trainingFeatureMatrix[key] = conversion.call(self.trainingDataFrame)

	def __generateTestFeatures(self):
		for key, conversion in self.featureTestGen.items():
			if not key in self.testFeatureMatrix.keys():
				# self.testFeatureMatrix[key] = self.preprocessor.createFeatureMatrix(featureSet)
				self.testFeatureMatrix[key] = conversion.call(self.testgDataFrame)

	def __addFeatureSet(self, name, trainingConversion, testConversion):
		self.featureTrainingGen[name] = trainingConversion
		self.featureTestGen[name] = testConversion

	def initClassifiers(self, trainingDf, testDf):
		self.trainingDataFrame = trainingDf
		self.testDataFrame = testDf

		self.__addFeatureSet('BOW', self.preprocessor.trainFeatureMatrix, self.preprocessor.createFeatureMatrix)
		self.__addFeatureSet('TextFeatures', self.textFeatureGenerator.calculate_features_with_dataframe, self.textFeatureGenerator.calculate_features_with_dataframe)

		self.__addClassifier("RandomForest", RandomForestBOWClassifier(self.preprocessor))
		self.__addClassifier("AdaBoost", AdaBoost(self.preprocessor))
		self.__addClassifier("Naive Bayes", BagOfWordsClassifier())

	def fitClassifiers(self, trainDf, groundTruth):
		self.__generateTrainingFeatures()
		self.__fitClassifiers()

	def testClassifiers(self, testDf, groundTruth):
		self.__generateTestFeatures()
		self.__testClassifiers()

	def getClassifierStatistics(self, featureSetName, classifierName):
		return self.classifiers[featureSetName][classifierName].testFeatureMatrix(None, None)
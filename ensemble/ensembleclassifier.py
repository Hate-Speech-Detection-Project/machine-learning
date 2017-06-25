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
import copy

class EnsembleClassifier:
	def __init__(self):
		self.threads = []
		self.preprocessor = Preprocessor()
		self.textFeatureGenerator = TextFeatureGenerator()

		self.scheduler = Scheduler()
		self.classifiers = {}

		self.trainingDataFrame = None
		self.testDataFrame = None
		self.trainingGroundTruth = None
		self.testGroundTruth = None

		self.featureSets = []

		self.featureTrainingGen = {}
		self.featureTestGen = {}

		# processed featuresets
		self.trainingFeatureMatrix = {}

		self.testFeatureMatrix = {}

	def __addClassifier(self, name, classifier):
		for featureSet in self.featureSets:
			if not featureSet in self.classifiers:
				self.classifiers[featureSet] = {}
			self.classifiers[featureSet][name] = copy.deepcopy(classifier)

	def __fitClassifiers(self):
		for featureSet in self.featureSets:
			for key, classifier in self.classifiers[featureSet].items():
				self.scheduler.schedule(function = classifier.fitFeatureMatrix, 
										args = (self.trainingFeatureMatrix[featureSet], 
												self.trainingGroundTruth))
		self.scheduler.joinAll()

	def __testClassifiers(self):
		for featureSet in self.featureSets:
			for key, classifier in self.classifiers[featureSet].items():
				self.scheduler.schedule(function = classifier.testFeatureMatrix, 
										args = (self.testFeatureMatrix[featureSet], 
												self.testGroundTruth))
		self.scheduler.joinAll();		

	def __generateTrainingFeatures(self):
		for key, conversion in self.featureTrainingGen.items():
			if not key in self.trainingFeatureMatrix.keys():
				# self.trainingFeatureMatrix[key] = self.preprocessor.trainFeatureMatrix(featureSet)
				self.trainingFeatureMatrix[key] = conversion(self.trainingDataFrame)
				print(key)
				print(self.trainingFeatureMatrix[key].shape)
				print(self.trainingFeatureMatrix[key])

	def __generateTestFeatures(self):
		for key, conversion in self.featureTestGen.items():
			if not key in self.testFeatureMatrix.keys():
				# self.testFeatureMatrix[key] = self.preprocessor.createFeatureMatrix(featureSet)
				self.testFeatureMatrix[key] = conversion(self.testDataFrame)

	def __addFeatureSet(self, name, trainingConversion, testConversion):
		self.featureSets.append(name)
		self.featureTrainingGen[name] = trainingConversion
		self.featureTestGen[name] = testConversion

	def initClassifiers(self, trainingDf, testDf, groundTruthName):
		self.trainingDataFrame = trainingDf
		self.trainingGroundTruth = trainingDf[groundTruthName]
		self.testDataFrame = testDf
		self.testGroundTruth = testDf[groundTruthName]

		self.__addFeatureSet('BOW', self.preprocessor.trainFeatureMatrix, self.preprocessor.createFeatureMatrix)
		self.__addFeatureSet('TextFeatures', self.textFeatureGenerator.calculate_features_with_dataframe, self.textFeatureGenerator.calculate_features_with_dataframe)

		self.__addClassifier("RandomForest", RandomForestBOWClassifier())
		self.__addClassifier("AdaBoost", AdaBoost(self.preprocessor))
		self.__addClassifier("Naive Bayes", BagOfWordsClassifier())

	def fitClassifiers(self):
		self.__generateTrainingFeatures()
		self.__fitClassifiers()

	def testClassifiers(self):
		self.__generateTestFeatures()
		self.__testClassifiers()

	def getClassifierStatistics(self, featureSetName, classifierName):
		return self.classifiers[featureSetName][classifierName].testFeatureMatrix(None, None)
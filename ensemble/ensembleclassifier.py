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

def identity(param):
	return param

class EnsembleClassifier:
	def __init__(self):
		self.threads = []
		self.preprocessor = Preprocessor()
		self.textFeatureGenerator = TextFeatureGenerator()

		self.scheduler = Scheduler()
		self.classifiers = {}

		self.defaultTrainingDataFrame = None
		self.defaultTestDataFrame = None

		self.testDataFrames = {}
		self.trainingDataFrames = {}

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

	def __fitClassifier(self, featureSet, classifier):
		self.scheduler.schedule(function = classifier.fitFeatureMatrix, 
						args = (self.trainingFeatureMatrix[featureSet], 
								self.trainingGroundTruth))

	def __fitClassifiers(self):
		for featureSet in self.featureSets:
			for key, classifier in self.classifiers[featureSet].items():
				__fitClassifier(featureset, classifier)
		self.scheduler.joinAll()

	def __testClassifier(self, featureSet, classifier):
		self.scheduler.schedule(function = classifier.testFeatureMatrix, 
						args = (self.testFeatureMatrix[featureSet], 
								self.testGroundTruth))

	def __testClassifiers(self):
		for featureSet in self.featureSets:
			for key, classifier in self.classifiers[featureSet].items():
				__testClassifier(featureset, classifier)
		self.scheduler.joinAll();		

	def __generateTrainingFeatures(self):
		for key, conversion in self.featureTrainingGen.items():
			if not key in self.trainingFeatureMatrix.keys():
				dataFrame = self.defaultTrainingDataFrame
				if key in trainingDataFrames:
					dataFrame = trainingDataFrames[key]
				self.trainingFeatureMatrix[key] = conversion(dataFrame)

	def __generateTestFeatures(self):
		for key, conversion in self.featureTestGen.items():
			if not key in self.testFeatureMatrix.keys():
				dataFrame = self.defaultTestDataFrame
				if key in testDataFrames:
					dataFrame = testDataFrames[key]
				self.testFeatureMatrix[key] = conversion(dataFrame)

	def __addFeatureSet(self, name, trainingConversion, testConversion, testDataFrame = None, trainingDataFrame = None):
		self.featureSets.append(name)
		self.featureTrainingGen[name] = trainingConversion
		self.featureTestGen[name] = testConversion
		self.testDataFrames[name] = testDataFrame
		self.trainingDataFrame[name] = trainingDataFrame

	def initClassifiers(self, defaultTrainingDf, defaultTestDf, ensembleTestDF, groundTruthName):
		self.defaultTrainingDataFrame = tdefaultTainingDf
		self.trainingGroundTruth = defaultTrainingDf[groundTruthName]
		self.defaultTestDataFrame = defaultTestDf
		self.testGroundTruth = defaultTestDf[groundTruthName]

		self.__addFeatureSet('BOW', self.preprocessor.trainFeatureMatrix, self.preprocessor.createFeatureMatrix)
		self.__addFeatureSet('BOW Ensemble Test', self.preprocessor.trainFeatureMatrix, self.preprocessor.createFeatureMatrix, ensembleTestDF)
		self.__addFeatureSet('TextFeatures', self.textFeatureGenerator.calculate_features_with_dataframe, self.textFeatureGenerator.calculate_features_with_dataframe)

		self.__addClassifier("RandomForest", RandomForestBOWClassifier())
		self.__addClassifier("AdaBoost", AdaBoost(self.preprocessor))
		self.__addClassifier("Naive Bayes", BagOfWordsClassifier())

	def initEnsembleClassifier(self, ensembleTestDF):
	    ensemble_training_data = np.matrix((getClassifierStatistics('BOW', 'RandomForest')[2],
                                    getClassifierStatistics('BOW', 'AdaBoost')[2],
                                    getClassifierStatistics('BOW', 'Naive Bayes')[2])).getT()
	    ensemble_test_data = np.matrix((getClassifierStatistics('BOW Ensemble Test', 'RandomForest')[2],
                                    getClassifierStatistics('BOW Ensemble Test', 'AdaBoost')[2],
                                    getClassifierStatistics('BOW Ensemble Test', 'Naive Bayes')[2])).getT()
	    __addFeatureSet("BOW Ensemble", identity, identity, ensemble_training_data, ensemble_test_data)
	    fitClassifiers()
	    testClassifiers()

	def fitClassifiers(self):
		self.__generateTrainingFeatures()
		self.__fitClassifiers()

	def testClassifiers(self):
		self.__generateTestFeatures()
		self.__testClassifiers()

	def getClassifierStatistics(self, featureSetName, classifierName):
		return self.classifiers[featureSetName][classifierName].testFeatureMatrix(None, None)
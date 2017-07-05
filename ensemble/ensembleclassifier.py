from bag_of_words import BagOfWordsClassifier
from random_forest import RandomForestBOWClassifier
from vote import Vote
from ada_boost import AdaBoost
from preprocessor import Preprocessor
from mpl_toolkits.mplot3d import Axes3D
from scheduler import Scheduler
import pandas as pd
import numpy as np
from utils import AnalysisInformation
from text_features import TextFeatureGenerator
from user_features import UserFeatureGenerator
import copy

def identity(param):
	return param

class EnsembleClassifier:
	def __init__(self):
		self.threads = []
		self.preprocessor = Preprocessor()
		self.textFeatureGenerator = TextFeatureGenerator()
		self.userFeatureGenerator = UserFeatureGenerator()

		self.scheduler = Scheduler()
		self.classifiers = {}

		self.defaultTrainingDataFrame = None
		self.defaultTestDataFrame = None

		self.testDataFrames = {}
		self.trainingDataFrames = {}

		self.trainingGroundTruth = None
		self.testGroundTruth = None

		self.featureSets = []
		self.classifierProtoTypes = {}

		self.featureTrainingGen = {}
		self.featureTestGen = {}

		# processed featuresets
		self.trainingFeatureMatrix = {}

		self.testFeatureMatrix = {}

	def __addClassifier(self, name, classifier):
		self.classifierProtoTypes[name] = classifier

	def __updateClassifiers(self):
		for featureSet in self.featureSets:
			if not featureSet in self.classifiers:
				self.classifiers[featureSet] = {}
			for key, classifier in self.classifierProtoTypes.items():
				if key not in self.classifiers[featureSet]:
					self.classifiers[featureSet][key] = copy.deepcopy(classifier)

	def __fitClassifier(self, featureSet, classifier, mode='parallel'):
		# Workaround, because the scikit random forest implementation is not thread-safe
		if mode is 'parallel':
			self.scheduler.schedule(function = classifier.fitFeatureMatrix, 
							args = (self.trainingFeatureMatrix[featureSet], 
									self.trainingGroundTruth))
		else:
			print("fittin classifier with:")
			print(featureSet)
			classifier.fitFeatureMatrix(self.trainingFeatureMatrix[featureSet], self.trainingGroundTruth)

	def __fitClassifiers(self):
		for featureSet in self.featureSets:
			for key, classifier in self.classifiers[featureSet].items():
				# Workaround, because the scikit random forest implementation is not thread-safe
<<<<<<< HEAD
				self.__fitClassifier(featureSet, classifier, 'single')
=======
#				if key is 'RandomForest':
					self.__fitClassifier(featureSet, classifier, 'single')
#				else:
#					self.__fitClassifier(featureSet, classifier)
>>>>>>> origin/scikit_confusion_matrix
		self.scheduler.joinAll()

	def __testClassifier(self, featureSet, classifier):
		self.scheduler.schedule(function = classifier.testFeatureMatrix, 
						args = (self.testFeatureMatrix[featureSet], 
								self.testGroundTruth))
		print("fittin classifier with:")
		print(featureSet)

	def __testClassifiers(self):
		for featureSet in self.featureSets:
			for key, classifier in self.classifiers[featureSet].items():
				self.__testClassifier(featureSet, classifier)
		self.scheduler.joinAll();		

	def __generateTrainingFeatures(self):
		for key, conversion in self.featureTrainingGen.items():
			if not key in self.trainingFeatureMatrix.keys():
				dataFrame = self.defaultTrainingDataFrame
				if key in self.trainingDataFrames and self.trainingDataFrames[key] is not None:
					dataFrame = self.trainingDataFrames[key]
				self.__prepareFeatureSet(self.trainingFeatureMatrix, key, conversion, dataFrame)
				# corpuses are not thread-safe :/
#				self.scheduler.schedule(function = self.__prepareFeatureSet, 
#										args = (key, conversion, dataFrame))
#		self.scheduler.joinAll();

	def __prepareFeatureSet(self, destination, key,  conversion, dataFrame):
		destination[key] = conversion(dataFrame)

	def __generateTestFeatures(self):
		for key, conversion in self.featureTestGen.items():
			if not key in self.testFeatureMatrix.keys():
				dataFrame = self.defaultTestDataFrame
				if key in self.testDataFrames and self.trainingDataFrames[key] is not None:
					dataFrame = self.testDataFrames[key]
				self.__prepareFeatureSet(self.testFeatureMatrix, key, conversion, dataFrame)

	def __addFeatureSet(self, name, trainingConversion, testConversion, testDataFrame = None, trainingDataFrame = None):
		self.featureSets.append(name)
		self.featureTrainingGen[name] = trainingConversion
		self.featureTestGen[name] = testConversion
		self.testDataFrames[name] = testDataFrame
		self.trainingDataFrames[name] = trainingDataFrame

	def initClassifiers(self, defaultTrainingDF, defaultTestDF, ensembleTestDF, groundTruthName):
		self.defaultTrainingDataFrame = defaultTrainingDF
		self.trainingGroundTruth = defaultTrainingDF[groundTruthName]
		self.defaultTestDataFrame = defaultTestDF
		self.testGroundTruth = defaultTestDF[groundTruthName]

		self.__addFeatureSet('BOW', self.preprocessor.trainFeatureMatrix, self.preprocessor.createFeatureMatrix)
		self.__addFeatureSet('BOW Ensemble Test', self.preprocessor.trainFeatureMatrix, self.preprocessor.createFeatureMatrix, ensembleTestDF)
		self.__addFeatureSet('TextFeatures', self.textFeatureGenerator.calculate_features_with_dataframe, self.textFeatureGenerator.calculate_features_with_dataframe)
		self.__addFeatureSet('UserFeatures', self.userFeatureGenerator.calculate_features_with_dataframe, self.userFeatureGenerator.calculate_features_with_dataframe)
		self.__addFeatureSet('UserFeatures Ensemble Test', self.userFeatureGenerator.calculate_features_with_dataframe, self.userFeatureGenerator.calculate_features_with_dataframe, ensembleTestDF)
		self.__addFeatureSet('TextFeatures Ensemble Test', self.textFeatureGenerator.calculate_features_with_dataframe, self.textFeatureGenerator.calculate_features_with_dataframe, ensembleTestDF)

		self.__addClassifier("RandomForest", RandomForestBOWClassifier())
		self.__addClassifier("AdaBoost", AdaBoost(self.preprocessor))
		self.__addClassifier("Naive Bayes", BagOfWordsClassifier())
		self.__updateClassifiers()

	def initEnsembleClassifier(self):
	    ensemble_training_data = np.matrix((
	    									self.getClassifierStatistics('BOW', 'RandomForest')[2],
		                                    self.getClassifierStatistics('BOW', 'AdaBoost')[2],
		                                    self.getClassifierStatistics('BOW', 'Naive Bayes')[2],
		                                    self.getClassifierStatistics('TextFeatures', 'RandomForest')[2],
		                                    self.getClassifierStatistics('TextFeatures', 'AdaBoost')[2],
		                                    self.getClassifierStatistics('TextFeatures', 'Naive Bayes')[2],
	    									self.getClassifierStatistics('UserFeatures', 'RandomForest')[2],
		                                    self.getClassifierStatistics('UserFeatures', 'AdaBoost')[2],
		                                    self.getClassifierStatistics('UserFeatures', 'Naive Bayes')[2])).getT()

	    ensemble_test_data = np.matrix((self.getClassifierStatistics('BOW Ensemble Test', 'RandomForest')[2],
	                                    self.getClassifierStatistics('BOW Ensemble Test', 'AdaBoost')[2],
	                                    self.getClassifierStatistics('BOW Ensemble Test', 'Naive Bayes')[2],
	                                    self.getClassifierStatistics('TextFeatures Ensemble Test', 'RandomForest')[2],
	                                    self.getClassifierStatistics('TextFeatures Ensemble Test', 'AdaBoost')[2],
	                                    self.getClassifierStatistics('TextFeatures Ensemble Test', 'Naive Bayes')[2],
	    								self.getClassifierStatistics('UserFeatures Ensemble Test', 'RandomForest')[2],
	                                    self.getClassifierStatistics('UserFeatures Ensemble Test', 'AdaBoost')[2],
	                                    self.getClassifierStatistics('UserFeatures Ensemble Test', 'Naive Bayes')[2])).getT()
	    self.__addFeatureSet("Ensemble", identity, identity, ensemble_training_data, ensemble_test_data)
	    self.__updateClassifiers()
	    self.fitClassifiers()
	    self.testClassifiers()

	def fitClassifiers(self):
		self.__generateTrainingFeatures()
		self.__fitClassifiers()

	def testClassifiers(self):
		self.__generateTestFeatures()
		self.__testClassifiers()

	def getClassifierStatistics(self, featureSetName, classifierName):
		return self.classifiers[featureSetName][classifierName].testFeatureMatrix(None, None)
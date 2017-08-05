from bag_of_words import BagOfWordsClassifier
from random_forest import RandomForestBOWClassifier
from svm import SVMClassifier
from vote import Vote
from ada_boost import AdaBoost
from preprocessor import Preprocessor
from mpl_toolkits.mplot3d import Axes3D
from scheduler import Scheduler
import pandas as pd
import numpy as np
from utils import AnalysisInformation, CorrelationMatrix
from text_features import TextFeatureGenerator
from user_features import UserFeatureGenerator
import copy

class EnsembleClassifier:
    def __init__(self):
        self.threads = []
        self.preprocessor = Preprocessor()
        self.ngramPreprocessor = Preprocessor((1,4))
        self.textFeatureGenerator = TextFeatureGenerator()
        self.userFeatureGenerator = UserFeatureGenerator()

        self.scheduler = Scheduler()
        self.classifiers = {}

        self.defaultTrainingDataFrame = None
        self.defaultTestDataFrame = None
        self.ensembleTestDataFrame = None
        self.defaultGroundTruthName = None

        self.testDataFrames = {}
        self.trainingDataFrames = {}

        self.trainingGroundTruth = None
        self.testGroundTruth = None 

        self.trainingGroundTruths = {}
        self.testGroundTruths = {}

        self.featureSets = []

        # featzuresets excluding ensembles
        self.baselineFeatureSets = []
        self.classifierProtoTypes = {}

        self.featureTrainingGen = {}
        self.featureTestGen = {}

        # processed featuresets
        self.trainingFeatureMatrix = {}

        self.testFeatureMatrix = {}
        self.correlationMatrix = None

    def getClassifierNames(self):
        return self.classifierProtoTypes.keys()

    def getFeatureSetNames(self):
        return self.featureSets

    def __addClassifier(self, name, classifier):
        self.classifierProtoTypes[name] = classifier

    def __updateClassifiers(self):
        for featureSet in self.featureSets:
            if not featureSet in self.classifiers:
                self.classifiers[featureSet] = {}
            for key, classifier in self.classifierProtoTypes.items():
                if key not in self.classifiers[featureSet]:
                    self.classifiers[featureSet][key] = copy.deepcopy(classifier)

    def __fitClassifier(self, featureSet, classifier, groundTruth, mode='parallel'):
        # Workaround, because the scikit random forest implementation is not thread-safe
        if mode is 'parallel':
           self.scheduler.schedule(function = classifier.fitFeatureMatrix, 
                           args = (self.trainingFeatureMatrix[featureSet], 
                                   groundTruth))
        else:
            self.scheduler.joinAll()
            classifier.fitFeatureMatrix(self.trainingFeatureMatrix[featureSet], groundTruth)

    def __fitClassifiers(self):
        for featureSet in self.featureSets:

            # fetch grountruth if not default
            groundTruth = self.trainingGroundTruth
            if featureSet in self.trainingGroundTruths.keys():
                groundTruth = self.trainingGroundTruths[featureSet]

            for key, classifier in self.classifiers[featureSet].items():
                # Workaround, because the scikit random forest implementation is not thread-safe
                if key is 'RandomForest':
                    self.__fitClassifier(featureSet, classifier, groundTruth, 'single')
                else:
                    self.__fitClassifier(featureSet, classifier, groundTruth)
        self.scheduler.joinAll()

    def __testClassifier(self, featureSet, classifier, groundTruth, mode='parallel'):
        if mode is 'parallel':
           self.scheduler.schedule(function = classifier.testFeatureMatrix, 
                           args = (self.testFeatureMatrix[featureSet], 
                                   self.testGroundTruth))
        else:
            self.scheduler.joinAll()
            classifier.testFeatureMatrix(self.testFeatureMatrix[featureSet], groundTruth)

    def __testClassifiers(self):
        for featureSet in self.featureSets:
            # fetch grountruth if not default
            groundTruth = self.testGroundTruth
            if featureSet in self.testGroundTruths.keys():
                groundTruth = self.testGroundTruths[featureSet]

            for key, classifier in self.classifiers[featureSet].items():
                # Workaround, because the scikit random forest implementation is not thread-safe
                if key is 'RandomForest':
                    self.__testClassifier(featureSet, classifier, groundTruth, 'single')
                else:
                    self.__testClassifier(featureSet, classifier, groundTruth)
        self.scheduler.joinAll();

    def testClassifiersSingle(self, comment, url):
        results = {}
        results['Reasons'] = {}
        for key, featureSet in enumerate(self.baselineFeatureSets):

            data = {
                'comment' : [comment],
                'created' : [1483457855],
                'url' : [url],
                'timestamp' : [1.483460e+09],
                'uid' : [0],
                'cid' : [0]
            }
            df = pd.DataFrame(data)

            generationFunction = self.featureTestGen[featureSet]
            # print(featureSet)
            # print(generationFunction)
            x = generationFunction(df)

            nb = self.classifiers['BOW']['Naive Bayes']

            for key, classifier in self.classifiers[featureSet].items():
                #print(featureSet + key)
                #print(x)
                results[featureSet + key] = classifier.predict(x)

        for tuple in nb.hate_words_and_indices:
            results['Reasons'][tuple[1]] = str(tuple[0])

        return results

    def __generateTrainingFeatures(self):
        for key, conversion in self.featureTrainingGen.items():
            if not key in self.trainingFeatureMatrix.keys():
                dataFrame = self.defaultTrainingDataFrame
                if key in self.trainingDataFrames and self.trainingDataFrames[key] is not None:
                    dataFrame = self.trainingDataFrames[key]
                self.__prepareFeatureSet(self.trainingFeatureMatrix, key, conversion, dataFrame)
                # corpuses are not thread-safe :/
                #self.scheduler.schedule(function = self.__prepareFeatureSet, 
                #                       args = (self.trainingFeatureMatrix, key, conversion, dataFrame))
        #self.scheduler.joinAll()

    def __prepareFeatureSet(self, destination, key,  conversion, dataFrame):
        destination[key] = conversion(dataFrame)

    def __generateTestFeatures(self):
        for key, conversion in self.featureTestGen.items():
            if not key in self.testFeatureMatrix.keys():
                dataFrame = self.defaultTestDataFrame
                if key in self.testDataFrames and self.testDataFrames[key] is not None:
                    dataFrame = self.testDataFrames[key]
                self.__prepareFeatureSet(self.testFeatureMatrix, key, conversion, dataFrame)

    def __addFeatureSet(self, name, trainingConversion, testConversion, testDataFrame = None, trainingDataFrame = None, groundTruthName = None):
        self.featureSets.append(name)
        self.baselineFeatureSets.append(name)
        self.featureTrainingGen[name] = trainingConversion
        self.featureTestGen[name] = testConversion

        if trainingDataFrame is not None:
            self.trainingDataFrames[name] = trainingDataFrame
            if groundTruthName is not None:
                #print(trainingDataFrame)
                #print(trainingDataFrame[groundTruthName].shape)
                self.trainingGroundTruths[name] = trainingDataFrame[groundTruthName]

        if testDataFrame is not None:
            self.testDataFrames[name] = testDataFrame
            if groundTruthName is not None:
                self.testGroundTruths[name] = testDataFrame[groundTruthName]

    def __addEnsembleFeatureSet(self, name, trainingConversion, testConversion, testDataFrame = None, trainingDataFrame = None, groundTruthName = None):
        self.featureSets.append(name)
        self.featureTrainingGen[name] = trainingConversion
        self.featureTestGen[name] = testConversion

        if trainingDataFrame is not None:
            self.trainingDataFrames[name] = trainingDataFrame
            if groundTruthName is not None:
                #print(trainingDataFrame)
                #print(trainingDataFrame[groundTruthName].shape)
                self.trainingGroundTruths[name] = trainingDataFrame[groundTruthName]

        if testDataFrame is not None:
            self.testDataFrames[name] = testDataFrame
            if groundTruthName is not None:
                self.testGroundTruths[name] = testDataFrame[groundTruthName]

    def initClassifiers(self, defaultTrainingDF, defaultTestDF, ensembleTestDF, groundTruthName):
        # print(defaultTrainingDF)

        self.defaultTrainingDataFrame = defaultTrainingDF
        self.trainingGroundTruth = defaultTrainingDF[groundTruthName]
        self.defaultTestDataFrame = defaultTestDF
        self.ensembleTestDataFrame = ensembleTestDF
        self.testGroundTruth = defaultTestDF[groundTruthName]
        self.defaultGroundTruthName = groundTruthName

        self.__addFeatureSet('BOW', self.preprocessor.trainFeatureMatrix, self.preprocessor.createFeatureMatrix)
        self.__addFeatureSet('BOWNGRAM', self.ngramPreprocessor.trainFeatureMatrix, self.ngramPreprocessor.createFeatureMatrix)
        self.__addEnsembleFeatureSet('BOWNGRAM Ensemble Test', self.preprocessor.trainFeatureMatrix, self.ngramPreprocessor.createFeatureMatrix, ensembleTestDF)
        self.__addEnsembleFeatureSet('BOW Ensemble Test', self.preprocessor.trainFeatureMatrix, self.preprocessor.createFeatureMatrix, ensembleTestDF)
        self.__addFeatureSet('TextFeatures', self.textFeatureGenerator.calculate_features_with_dataframe, self.textFeatureGenerator.calculate_features_with_dataframe)
        self.__addFeatureSet('UserFeatures', self.userFeatureGenerator.calculate_features_with_dataframe, self.userFeatureGenerator.calculate_features_with_dataframe)
        self.__addEnsembleFeatureSet('UserFeatures Ensemble Test', self.userFeatureGenerator.calculate_features_with_dataframe, self.userFeatureGenerator.calculate_features_with_dataframe, ensembleTestDF, groundTruthName = groundTruthName)
        self.__addEnsembleFeatureSet('TextFeatures Ensemble Test', self.textFeatureGenerator.calculate_features_with_dataframe, self.textFeatureGenerator.calculate_features_with_dataframe, ensembleTestDF, groundTruthName = groundTruthName)

        self.__addClassifier("AdaBoost", AdaBoost(self.preprocessor))
        self.__addClassifier("Naive Bayes", BagOfWordsClassifier())
        self.__addClassifier("RandomForest", RandomForestBOWClassifier())
        self.__addClassifier("SVM", SVMClassifier())
        self.__updateClassifiers()

    def initEnsembleClassifier(self):
        ensemble_training_data = {}
        classifiers = self.getSparslyCorrelatingClassifiers(blackList = 'Ensemble')
        print(classifiers)
        for classifier in classifiers:
            ensemble_training_data[classifier[0] + ' ' +  classifier[1]] = self.getClassifierStatistics(classifier[0], classifier[1])[2]
        ensemble_training_data[self.defaultGroundTruthName] = self.defaultTestDataFrame[self.defaultGroundTruthName]
        #print(ensemble_training_data)
        ensembleTrainingDF = pd.DataFrame(data = ensemble_training_data)

        ensemble_test_data = {}
        for classifier in classifiers:
            ensemble_test_data[classifier[0] + ' ' + classifier[1]] = self.getClassifierStatistics(classifier[0] + ' Ensemble Test', classifier[1])[2]
        ensemble_test_data[self.defaultGroundTruthName] = self.ensembleTestDataFrame[self.defaultGroundTruthName]
        for key, classifier in ensemble_test_data.items():
            print(classifier.shape)
        ensembleTestDF = pd.DataFrame(data = ensemble_test_data)

        self.__addEnsembleFeatureSet("Ensemble", self.extractGroundTruth, self.extractGroundTruth, ensembleTrainingDF, ensembleTestDF)
        self.__updateClassifiers()
        self.fitClassifiers()
        self.testClassifiers()

        nb = self.classifiers['BOW']['Naive Bayes']
        nb.feature_names = self.preprocessor.feature_names

    def getCorrelationMatrix(self):
        if self.correlationMatrix is None:
            dataRows = {}
            for featureSetName in self.getFeatureSetNames():
                for classifierName in self.getClassifierNames():
                    dataRows[featureSetName[:1] + classifierName] = self.getClassifierStatistics(featureSetName, classifierName)[2]

            self.correlationMatrix = CorrelationMatrix(dataRows)

        return self.correlationMatrix

    def getCompleteTrainingSet(self):
        joined_data = {}
        
        for key, dataSet in self.trainingDataFrames.items():
            joined_data[key] = dataSet

        return pd.DataFrame(data = joined_data)

    def getSparslyCorrelatingClassifiers(self, blackList = None, whiteList = None):
        sparselyCorrelatingCombinations = set()
        possibleCombinations = set()

        for featureSet in self.featureSets:
            if blackList is not None and blackList in featureSet: 
                continue
            if whiteList is not None and whiteList not in featureSet: 
                continue
            for key, classifier in self.classifierProtoTypes.items():
                possibleCombinations.add((featureSet, key))

        for combination0 in possibleCombinations:
            for combination1 in possibleCombinations:
                dataRow0 = self.getClassifierStatistics(combination0[0], combination0[1])[2]
                dataRow1 = self.getClassifierStatistics(combination1[0], combination1[1])[2]
                correlation = np.corrcoef(dataRow0, dataRow1)[0, 1]
                if(correlation <= 0.2):
                    sparselyCorrelatingCombinations.add(combination0)
                    sparselyCorrelatingCombinations.add(combination1)

        return sparselyCorrelatingCombinations

    def fitClassifiers(self):
        self.__generateTrainingFeatures()
        self.__fitClassifiers()

    def testClassifiers(self):
        self.__generateTestFeatures()
        self.__testClassifiers()

    def getClassifierStatistics(self, featureSetName, classifierName):
        return self.classifiers[featureSetName][classifierName].testFeatureMatrix(None, None)

    def extractGroundTruth(self, dataFrameWithGroundTruth):
        # print(dataFrameWithGroundTruth.as_matrix())
        featureMatrix = dataFrameWithGroundTruth.drop(self.defaultGroundTruthName, axis=1).as_matrix()
        return featureMatrix

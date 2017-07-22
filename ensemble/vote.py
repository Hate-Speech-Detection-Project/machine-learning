from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from preprocessor import Preprocessor
from utils import AnalysisInformation

from preprocessor import Preprocessor

class Vote:
    def __init__(self, threshold, verbose = False):
        self.results = []
        self.threshold = threshold
        self.verbose = verbose

    def fitFeatureMatrix(self, x):
        self.results = [None] * len(x)
        columncount = len(x[0])

        for index, row in enumerate(x):
            rowvalue = 0;
            rowvalue = np.sum(row)
            if(rowvalue >= self.threshold):
                self.results[index] = 1
            else:
                self.results[index] = 0

    def getResults(self, result):

        #equals = 0
        #for index, row in enumerate(self.results):
        #    if(row == result[index]):
        #        equals += 1

        if self.verbose:
            print(self.results)
            print(result)

        analysisInformation = AnalysisInformation(self.results, result, self.verbose)
        return (analysisInformation, self.results)

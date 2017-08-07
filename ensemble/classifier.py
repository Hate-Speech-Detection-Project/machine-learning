import numpy as np

class Classifier:
	def getWeights(self, labelArray):
	    unique, counts = np.unique(labelArray, return_counts=True)
	    classCounts = dict(zip(unique, counts))
	    weights = [None] * len(labelArray)

	    for key, label in enumerate(labelArray):
	        weights[key] = len(labelArray)/classCounts[label]

	    return np.array(weights)
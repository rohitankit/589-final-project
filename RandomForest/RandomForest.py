from .DecisionTree import DecisionTree

from collections import defaultdict
import random

class RandomForest:

    def __init__(self, k):
        self.k = k
        self.ensembleTrees = []
        for treeIdx in range(k):
            self.ensembleTrees.append(DecisionTree())
        
    def train(self, trainingData, attributeTypes, labelIdx):
        for decisionTree in self.ensembleTrees:
            trainingBootstrap = self.getBootstrap(trainingData)
            decisionTree.train(trainingBootstrap, attributeTypes, labelIdx)
    
    def getBootstrap(self, trainingData):
        bootstrapSize = len(trainingData)
        bootstrapSet = []

        for idx in range(bootstrapSize):
            bootstrapSet.append(random.choice(trainingData))
        
        return bootstrapSet

    def predict(self, testInstance):
        predictionFreq = defaultdict(lambda :0)
        for tree in self.ensembleTrees:
            predictionFreq[tree.predict(testInstance)] += 1
        
        majorityLabel = None
        maxFreq = 0
        for label, freq in predictionFreq.items():
            if freq > maxFreq:
                majorityLabel = label
                maxFreq = freq
        
        return majorityLabel
import math
import random
from collections import defaultdict
from .LeafNode import LeafNode
from .TreeNode import TreeNode

class TreeCreator:
    def __init__(self, maxDepth):
        self.attributesToSplit = []
        self.trainingInstances = []
        self.attributeTypes = []
        self.labelIdx = -1
        self.maxDepth = maxDepth
    
    def create(self, attributesToSplit, trainingInstances, attributeTypes, labelIdx):
        self.attributesToSplit = attributesToSplit
        self.trainingInstances = trainingInstances
        self.attributeTypes = attributeTypes
        self.labelIdx = labelIdx

        if self.isLeaf():
            return LeafNode(self.trainingInstances, self.labelIdx)

        bestSplitIdx = self.getOptimalSplit()

        return TreeNode(bestSplitIdx, self.attributesToSplit, self.trainingInstances, self.attributeTypes, self.labelIdx, self.maxDepth)
    
    def isLeaf(self):
        if self.maxDepth <= 0:
            return True
        
        NodeLabel = self.trainingInstances[0][self.labelIdx]
        for instance in self.trainingInstances:
            instanceLabel = instance[self.labelIdx]
            if instanceLabel != NodeLabel:
                return False
        return True
    
    def getOptimalSplit(self):
        originalEntropy = self.getEntropy(self.trainingInstances)
        maxInfoGain = 0
        bestSplitIdx = -1

        availableAttributes = TreeCreator.getAvailableAttributes(self.attributesToSplit.copy())

        for splitIdx in availableAttributes:

            newEntropy = 0
            partitions = self.getPartitions(self.trainingInstances, splitIdx, self.attributeTypes[splitIdx])
            
            for splitValue, instances in partitions.items():
                newEntropy += (len(instances)/len(self.trainingInstances)) * self.getEntropy(instances)
        
            infoGain = originalEntropy - newEntropy
            if infoGain > maxInfoGain:
                maxInfoGain = infoGain
                bestSplitIdx = splitIdx
        
        return bestSplitIdx

    def getOptimalSplitGini(self):
        minGini = 1
        bestSplitIdx = -1
        availableAttributes = TreeCreator.getAvailableAttributes(self.attributesToSplit.copy())

        for splitIdx in availableAttributes:
            newEntropy = 0
            partitions = self.getPartitions(self.trainingInstances, splitIdx, self.attributeTypes[splitIdx])
            
            for splitValue, instances in partitions.items():
                newEntropy += (len(instances)/len(self.trainingInstances)) * self.getGini(instances)
        
            if newEntropy <= minGini:
                minGini = newEntropy
                bestSplitIdx = splitIdx
        
        return bestSplitIdx

    def getAvailableAttributes(attributesToSplit):
        prunedAttributesLength = math.ceil(len(attributesToSplit) ** (0.5))

        availableAttributes = []
        for i in range(prunedAttributesLength):
            chosenAttribute = random.choice(attributesToSplit)
            availableAttributes.append(chosenAttribute)
            attributesToSplit.remove(chosenAttribute)
        
        return availableAttributes

    def getEntropy(self, dataset):
        labelFreq = defaultdict(lambda :0)
        datasetLen = len(dataset)

        for instance in dataset:
            instanceLabel = instance[self.labelIdx]
            labelFreq[instanceLabel] += 1
        
        entropy = 0
        for freq in labelFreq.values():
            entropy += (-freq/datasetLen) * math.log2(freq/datasetLen)
        return entropy

    def getGini(self, dataset):
        labelFreq = defaultdict(lambda :0)
        datasetLen = len(dataset)

        for instance in dataset:
            instanceLabel = instance[self.labelIdx]
            labelFreq[instanceLabel] += 1
        
        gini = 1
        for freq in labelFreq.values():
            gini -= (freq/datasetLen)**2
        return gini
    
    def getPartitions(self, dataset, splitIdx, splitType):
        partitions = defaultdict(lambda :[])
        avgFeatureValue = TreeCreator.getAverage(dataset, splitIdx)

        for instance in dataset:
            splitValue = instance[splitIdx]
            if splitType:
                partitions[splitValue].append(instance)
            else:
                if splitValue >= avgFeatureValue:
                    partitions["greater"].append(instance)
                else:
                    partitions["less"].append(instance)
                    
        return partitions
    
    def getAverage(dataset, splitIdx):
        attributeSum = 0
        for instance in dataset:
            attributeSum += instance[splitIdx]
        
        return attributeSum/len(dataset)
    
    def emptyPartitionExists(self, partitions):
        return len(partitions.keys()) < 3

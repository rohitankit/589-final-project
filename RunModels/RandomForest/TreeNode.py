from collections import defaultdict
from . import TreeCreator

class TreeNode:

    def __init__(self, splitIdx, attributesToSplit, trainingInstances, attributeTypes, labelIdx, maxDepth):
        self.splitIdx = splitIdx
        self.attributeTypes = attributeTypes
        self.trainingInstances = trainingInstances
        self.labelIdx = labelIdx
        self.childNodes = {}
        self.partitionValue = 0

        splitType = attributeTypes[splitIdx]
        partitions = self.getPartitions(trainingInstances, self.splitIdx, splitType)
        for splitValue, instances in partitions.items():
            subtree = TreeCreator.TreeCreator(maxDepth-1)
            self.childNodes[splitValue] = subtree.create(attributesToSplit, instances, attributeTypes, labelIdx)
        
    def predict(self, testInstance):
        splitType = self.attributeTypes[self.splitIdx]
        splitValue = testInstance[self.splitIdx]

        if splitType:
            if splitValue not in self.childNodes.keys():
                return self.getMajorityLabel()
            childPath = self.childNodes[splitValue]
        else:
            if splitValue >= self.partitionValue:
                childPath = self.childNodes["greater"]
            else:
                childPath = self.childNodes["less"]

        return childPath.predict(testInstance)

    def getPartitions(self, dataset, splitIdx, splitType):
        partitions = defaultdict(lambda :[])
        self.partitionValue = TreeNode.getAverage(dataset, splitIdx)

        for instance in dataset:
            splitValue = instance[splitIdx]
            if splitType:
                partitions[splitValue].append(instance)
            else: 
                if splitValue >= self.partitionValue:
                    partitions["greater"].append(instance)
                else:
                    partitions["less"].append(instance)
                    
        return partitions
    
    def getAverage(dataset, splitIdx):
        attributeSum = 0
        for instance in dataset:
            attributeSum += instance[splitIdx]
        
        return attributeSum/len(dataset)

    def getMajorityLabel(self):
        labelFreq = defaultdict(lambda :0)
        for instance in self.trainingInstances:
            instanceLabel = instance[self.labelIdx]
            labelFreq[instanceLabel] += 1
        
        maxFreq = -1
        majorityLabel = 0
        for label, freq in labelFreq.items():
            if freq > maxFreq:
                maxFreq = freq
                majorityLabel = label
        
        return majorityLabel
        
    
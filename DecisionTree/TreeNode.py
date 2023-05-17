from collections import defaultdict
from . import TreeCreator

class TreeNode:

    def __init__(self, splitIdx, attributesToSplit, trainingInstances):
        self.trainingInstances = trainingInstances
        self.splitIdx = splitIdx
        self.childNodes = {}
        self.LABEL_IDX = len(trainingInstances[0])-1

        partitions = self.getPartitions(trainingInstances, self.splitIdx)
        for splitValue, instances in partitions.items():
            subtree = TreeCreator.TreeCreator(attributesToSplit, instances)
            self.childNodes[splitValue] = subtree.create()
        
    def predict(self, testInstance):
        splitValue = testInstance[self.splitIdx]
        
        if splitValue not in self.childNodes.keys():
            return self.getMajorityLabel()
        else:
            childPath = self.childNodes[splitValue]
            return childPath.predict(testInstance)

    def getPartitions(self, dataset, splitIdx):
        partitions = defaultdict(lambda :[])
        for instance in dataset:
            splitValue = instance[splitIdx]
            partitions[splitValue].append(instance)
        return partitions

        # partitions = defaultdict(lambda :[])
        # avgFeatureValue = TreeCreator.getAverage(dataset, splitIdx)

        # for instance in dataset:
        #     splitValue = instance[splitIdx]
        #     if splitValue >= avgFeatureValue:
        #         partitions["greater"].append(instance)
        #     else:
        #         partitions["less"].append(instance)
                    
        # return partitions
    
    def getMajorityLabel(self):
        labelFreq = defaultdict(lambda :0)
        for instance in self.trainingInstances:
            instanceLabel = instance[self.LABEL_IDX]
            labelFreq[instanceLabel] += 1
        
        maxFreq = -1
        majorityLabel = 0
        for label, freq in labelFreq.items():
            if freq > maxFreq:
                maxFreq = freq
                majorityLabel = label
        
        return majorityLabel
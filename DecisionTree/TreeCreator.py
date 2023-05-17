import math
from collections import defaultdict
from .LeafNode import LeafNode
from .TreeNode import TreeNode

class TreeCreator:

    def __init__(self, attributesToSplit, trainingInstances):
        self.attributesToSplit = attributesToSplit
        self.trainingInstances = trainingInstances
        self.LABEL_IDX = len(trainingInstances[0])-1
    
    def create(self):
        if self.isLeaf():
            return LeafNode(self.trainingInstances)

        bestSplitIdx = self.getOptimalSplit()
        # print(self.trainingInstances)
        # print(bestSplitIdx)
        splitPartitions = self.getPartitions(self.trainingInstances, bestSplitIdx)
        
        if self.emptyPartitionExists(splitPartitions) or bestSplitIdx == -1:
            return LeafNode(self.trainingInstances)
        
        else:
            self.attributesToSplit.remove(bestSplitIdx)
            return TreeNode(bestSplitIdx, self.attributesToSplit, self.trainingInstances)
    
    def isLeaf(self):
        if len(self.attributesToSplit) == 0:
            return True
        
        NodeLabel = self.trainingInstances[0][self.LABEL_IDX]
        for instance in self.trainingInstances:
            instanceLabel = instance[self.LABEL_IDX]
            if instanceLabel != NodeLabel:
                return False
        return True
    
    def getOptimalSplit(self):
        originalEntropy = self.getEntropy(self.trainingInstances)
        maxInfoGain = 0
        bestSplitIdx = -1 

        for splitIdx in self.attributesToSplit:
            newEntropy = 0
            partitions = self.getPartitions(self.trainingInstances, splitIdx)
            for splitValue, instances in partitions.items():
                newEntropy += (len(instances)/len(self.trainingInstances)) * self.getEntropy(instances)
        
            infoGain = originalEntropy - newEntropy
            if infoGain >= maxInfoGain:
                maxInfoGain = infoGain
                bestSplitIdx = splitIdx
            
            print(splitIdx, list(map(lambda x: len(x), partitions.values())), infoGain)
        
        print(bestSplitIdx)
        print("\n")

        return bestSplitIdx

    def getEntropy(self, dataset):
        labelFreq = defaultdict(lambda :0)
        datasetLen = len(dataset)

        for instance in dataset:
            instanceLabel = instance[self.LABEL_IDX]
            labelFreq[instanceLabel] += 1
        
        entropy = 0
        for freq in labelFreq.values():
            entropy += (-freq/datasetLen) * math.log2(freq/datasetLen)
        return entropy
    
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

    # def getAverage(dataset, splitIdx):
    #     attributeSum = 0
    #     for instance in dataset:
    #         attributeSum += instance[splitIdx]
        
    #     return attributeSum/len(dataset)
    
    def emptyPartitionExists(self, partitions):
        return len(partitions.keys()) < 3

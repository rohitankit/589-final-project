import csv
import random
import math
from collections import defaultdict
import numpy as np

class Dataset:

    def __init__(self):
        self.featureLen = 0
        self.datasetInstances = []
        self.datasetLabels = []

        self._datasetClassPartition = []
    
    def load(self, dataInstances):
        self.datasetInstances = dataInstances
        self.featureLen = len(dataInstances[0])-1
        
        classInstancesDict = defaultdict(lambda :[])
        for idx, instance in enumerate(self.datasetInstances):
            classInstancesDict[instance[-1]].append(instance)
        
        for classInstances in classInstancesDict.values():
            self._datasetClassPartition.append(classInstances)
    
    def _parse(self):
        """
        Parses dataFile and seperates instances into training and testing set

        returns: None
        """
        classInstancesDict = defaultdict(lambda :[])

        if self.dataFile:
            with open(self.dataFile, "r") as f:
                reader = csv.reader(f, delimiter=self.delimiter)

                for i, line in enumerate(reader):
                    if i == 0:
                        continue
                    if len(line) > 0:
                        parsedInstance = []
                        for feature in line:
                            parsedInstance.append(float(feature))
                        self._dataset.append(parsedInstance)
                        
                        instanceClass = parsedInstance[self.classIdx]
                        classInstancesDict[instanceClass].append(parsedInstance)
        
        for classInstances in classInstancesDict.values():
            self._datasetClassPartition.append(classInstances)

    def getDataset(self):
        return self._dataset

    def getKFoldPartitions(self, k):
        """
        splits dataset into training and test data

        returns: None
        """
        kFoldDataset = [[] for _ in range(k)]

        for classInstances in self._datasetClassPartition:
            random.shuffle(classInstances)
            classInstancePartitions = Dataset._partition(classInstances, k)
            
            for foldIdx in range(len(classInstancePartitions)):
                partitionInstances = classInstancePartitions[foldIdx]
                kFoldDataset[foldIdx] += partitionInstances
        
        for i in range(k):
            random.shuffle(kFoldDataset[i])
        
        return kFoldDataset
    
    def _partition(lst, k): 
        division = len(lst) / float(k) 
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(k) ]
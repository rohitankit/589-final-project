import csv
import random
import math
from collections import defaultdict
import numpy as np

class Dataset:

    def __init__(self):
        self.featureLen = 0
        self.datasetInstances = []

        self._datasetClassPartition = []
    
    def load(self, dataInstances):
        """
        loads dataInstances to Dataset object

        returns: None
        """
        self.datasetInstances = dataInstances
        self.featureLen = len(dataInstances[0])-1
        
        classInstancesDict = defaultdict(lambda :[])
        for idx, instance in enumerate(self.datasetInstances):
            classInstancesDict[instance[-1]].append(instance)
        
        for classInstances in classInstancesDict.values():
            self._datasetClassPartition.append(classInstances)
    
    def loadFromFile(self, file, delimiter, classIdx, ignoreAttributes):
        """
        Parses dataFile and seperates instances into training and testing set

        returns: None
        """
        classInstancesDict = defaultdict(lambda :[])

        if file:
            with open(file, "r") as f:
                reader = csv.reader(f, delimiter=delimiter)

                for i, line in enumerate(reader):
                    if i == 0:
                        continue
                    if len(line) > 0:
                        parsedInstance = []
                        for idx, feature in enumerate(line):
                            if idx not in ignoreAttributes:
                                if not Dataset.is_float(feature):
                                    ascii_sum = 0
                                    for char in feature:
                                        ascii_sum += ord(char)
                                    parsedInstance.append(ascii_sum) 
                                else:   
                                    parsedInstance.append(float(feature))
                        self.datasetInstances.append(parsedInstance)
                        
                        instanceClass = parsedInstance[classIdx]
                        classInstancesDict[instanceClass].append(parsedInstance)
        
        for classInstances in classInstancesDict.values():
            self._datasetClassPartition.append(classInstances)
    
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

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
    
    def oneHotEncodeLabel(self, datasetLabels, classIdx):

        self._datasetClassPartition = []
        newDatasetInstances = []

        classInstancesDict = defaultdict(lambda : [])

        for instance in self.datasetInstances:
            classValue = instance[classIdx]
            encodedInstance = instance[:classIdx] + [datasetLabels.index(classValue)] + instance[classIdx+1:]
            newDatasetInstances.append(encodedInstance)

            instanceClass = encodedInstance[classIdx]
            classInstancesDict[instanceClass].append(encodedInstance)
            
        for classInstances in classInstancesDict.values():
            self._datasetClassPartition.append(classInstances)
        
        self.datasetInstances = newDatasetInstances

    def getRawDataset(self):
        return self.datasetInstances
    
    def _partition(lst, k): 
        division = len(lst) / float(k) 
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(k) ]
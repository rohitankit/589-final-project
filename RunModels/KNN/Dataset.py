import csv
import random
import math
from sklearn.utils import shuffle

class Dataset:

    def __init__(self, dataFile):
        """
        Constructor for dataset class

        :param dataFile: csv file containing data instances

        returns: Dataset Object
        """
        self.dataFile = dataFile
        self._dataset = []
        self._trainingData, self._testData = [], []
        
        self._parse()
        self.shuffle()
        self.getBounds()

    def load(self, dataFile):
        """
        Loads a new CSV file into the dataset

        :param dataFile: csv file containing data instances

        returns: None
        """
        self.dataFile = dataFile
        self._parse()
        self.shuffle()
        self.getBounds()
    
    def _parse(self):
        """
        Parses dataFile and seperates instances into training and testing set

        returns: None
        """
        if self.dataFile:
            with open(self.dataFile, "r") as f:
                reader = csv.reader(f, delimiter="\t")

                for i, line in enumerate(reader):
                    if len(line) > 0:
                        rawInstance = line[0].split(',')
                        parsedInstance = [float(rawInstance[0]), float(rawInstance[1]), float(rawInstance[2]), float(rawInstance[3]), rawInstance[4]]
                        self._dataset.append(parsedInstance)
    
    def getBounds(self):

        sepalLengthBounds = [float('inf'), -float('inf')]
        sepalWidthBounds = [float('inf'), -float('inf')]
        petalLengthBounds = [float('inf'), -float('inf')]
        petalWidthBounds = [float('inf'), -float('inf')]

        for sepalLength, sepalWidth, petalLength, petalWidth, _ in self._trainingData:
            sepalLengthBounds = Dataset.addToBounds(sepalLengthBounds, sepalLength)
            sepalWidthBounds = Dataset.addToBounds(sepalWidthBounds, sepalWidth)
            petalLengthBounds = Dataset.addToBounds(petalLengthBounds, petalLength)
            petalWidthBounds = Dataset.addToBounds(petalWidthBounds, petalWidth)
        
        self._trainingData = self._normalize(self._trainingData, sepalLengthBounds, sepalWidthBounds, petalLengthBounds, petalWidthBounds)
        self._testData = self._normalize(self._testData, sepalLengthBounds, sepalWidthBounds, petalLengthBounds, petalWidthBounds)

    def _normalize(self, dataset, sepalLengthBounds, sepalWidthBounds, petalLengthBounds, petalWidthBounds):
        """
        Normalizes all features of the array between 0 and 1

        returns: None
        """
        normalizedData = []
        
        for sepalLength, sepalWidth, petalLength, petalWidth, label in dataset:
            normalizedSepalLength = (sepalLength - sepalLengthBounds[0]) / (sepalLengthBounds[1] - sepalLengthBounds[0])
            normalizedSepalWidth = (sepalWidth - sepalWidthBounds[0]) / (sepalWidthBounds[1] - sepalWidthBounds[0])
            normalizedPetalLength = (petalLength - petalLengthBounds[0]) / (petalLengthBounds[1] - petalLengthBounds[0])
            normalizedPetalWidth = (petalWidth - petalWidthBounds[0]) / (petalWidthBounds[1] - petalWidthBounds[0])
            normalizedData.append([normalizedSepalLength, normalizedSepalWidth, normalizedPetalLength, normalizedPetalWidth, label])
        
        return normalizedData


    def addToBounds(oldBound, newValue):
        """
        Incorporates a new instance to the boundary of a feauture
        """
        minOldBound, maxOldBound = oldBound
        return [min(minOldBound, newValue), max(maxOldBound, newValue)]


    def _partition(self):
        """
        splits dataset into training and test data

        returns: None
        """

        if self._dataset:
            trainingSize = round(0.8 * len(self._dataset))
            testSize = len(self._dataset)-trainingSize

            partitionStart = random.randint(0, len(self._dataset)-testSize)
            partitionEnd = partitionStart+testSize
            
            self._testData = self._dataset[partitionStart:partitionEnd]
            self._trainingData = self._dataset[:partitionStart] + self._dataset[partitionEnd:len(self._dataset)]
        
    
    def shuffle(self):
        """
        Shuffles dataset and reconfigures training and test data

        returns: None
        """
        self._dataset = shuffle(self._dataset, random_state=0)
        self._partition()
    
    def getTrainingData(self):
        """
        returns Training data

        returns: []
        """
        return self._trainingData
    
    def getTestData(self):
        """
        returns Test data

        returns: []
        """
        return self._testData

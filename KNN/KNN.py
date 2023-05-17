from collections import defaultdict
from statistics import mean
from .ModelEvaluation import ModelEvaluation

class KNN():
    
    def __init__(self, k, trainingData, testData):
        """
        Constructor for KNN class

        :param k: Hyperparameter of algorithm
        :param trainingData: Array of data instances that the algorithm will be trained on
        :param testData: Array of data instances that the algorithm will be tested on

        returns: KNN Object
        """
        self.k = k
        self.trainingData = trainingData
        self.testData = testData
    
    def loadDataset(self, trainingData, testData):
        """
        Loads training and test data for the model

        :param trainingData: Array of data instances that the algorithm will be trained on
        :param testData: Array of data instances that the algorithm will be tested on

        returns: None
        """
        self.trainingData = trainingData
        self.testData = testData
    
    def predict(self, testInstance, classIdx):
        """
        predicts the label of a test Instance of features by averaging k Nearest Neighbors in space

        :param testInstance: Data Instance for which the label will be predicted 

        returns: Label for the Test Instance
        """
        sortedInstances = []

        for idx in range(len(self.trainingData)):
            trainingInstance = self.trainingData[idx]
            trainingAttributes = [trainingInstance[:classIdx], trainingInstance[classIdx+1:]]
            trainingAttributes = [instance for fold in trainingAttributes for instance in fold]

            testAttributes = [testInstance[:classIdx], testInstance[classIdx+1:]]
            testAttributes = [instance for fold in testAttributes for instance in fold]

            instanceDifference = KNN.getEuclidianDist(trainingAttributes, testAttributes)
            sortedInstances.append((instanceDifference, trainingInstance[classIdx]))

        sortedInstances.sort(key = lambda x:x[0])

        kClosestInstances = []
        for i in range(self.k):
            kClosestInstances.append(sortedInstances[i])
        
        labelFrequency = defaultdict(lambda :0) 
        for _, label in kClosestInstances:
            labelFrequency[label] += 1

        majorityLabel, maxFreq = "", 0 
        for label, frequency in labelFrequency.items():
            if frequency > maxFreq:
                maxFreq = frequency
                majorityLabel = label

        return majorityLabel
    
    def getEuclidianDist(trainingInstance, testInstance):
        instanceDifference = 0
        for featureIdx in range(len(trainingInstance)):
            trainingFeature = trainingInstance[featureIdx]
            testFeature = testInstance[featureIdx]
            featureDifference = (trainingFeature - testFeature) ** 2
            instanceDifference += featureDifference

        return instanceDifference ** (1/2)
    
    def getTestAccuracy(self):
        correctPredictions = 0
        for testInstance in self.testData:
            predictedLabel = self.predict(testInstance)
            actualLabel = testInstance[-1]
            
            if actualLabel == predictedLabel:
                correctPredictions += 1
        
        return (correctPredictions/len(self.testData))
    
    def getTrainingAccuracy(self):
        correctPredictions = 0
        a, b = 0, 0
        for trainingInstance in self.trainingData:
            predictedLabel = self.predict(trainingInstance)
            actualLabel = trainingInstance[-1]
            
            # print(predictedLabel, actualLabel)
            if actualLabel == predictedLabel:
                correctPredictions += 1
            else:
                a += 1
                if trainingInstance in self.trainingData:
                    b+=1

        print(a, b)
        return (correctPredictions/len(self.trainingData))

    def evaluate(self, datasetLabels, classIdx):
        modelEvaluator = ModelEvaluation()
        modelEvaluator.setLabels(datasetLabels)

        for testInstance in self.testData:
            predictedClass = self.predict(testInstance, classIdx)
            actualClass = testInstance[classIdx]

            modelEvaluator.addInstanceEvaluation(int(actualClass), int(predictedClass))

        return modelEvaluator.getAccuracy(), modelEvaluator.getF1Score()
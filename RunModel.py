from LoadData import LoadData
from KNN.KNN import KNN
from DecisionTree.DecisionTree import DecisionTree
from RandomForest.Main import EnsembleEvaluation

class RunModels:

    def __init__(self):
        self.kFoldPartitions = []

    def runDigitsModels(self):
        digitsData = LoadData()
        dataset, self.kFoldPartitions = digitsData.loadDigitDataset()
        digitsAttributeTypes = [False]*(len(dataset[0])-1) + [True]

        datasetLabels = set()
        for instance in dataset:
            datasetLabels.add(instance[-1])
        datasetLabels = list(datasetLabels)
        
        KNN_Accuracies = []
        KNN_F1_Scores = []
        DecisionTreeAccuracy = []

        # randomForestHyperParameters = [5, 10, 15, 20]

        # print("Dataset 1 metric evaluation using random forests\n")
        # print("  k-value  |  Accuracy  | F1 Score ")
        # print("------------------------------------")

        # for k in randomForestHyperParameters:
        #     randomForestModel = EnsembleEvaluation(k)
        #     randomForestModel.initDataset(dataset, self.kFoldPartitions, digitsAttributeTypes)
        #     randomForestAccuracy, randomForestF1 = randomForestModel.evaluate()

        #     print("  {:.4f}  |   {:.4f}   |  {:.4f}".format(k, randomForestAccuracy, randomForestF1))

        # randomForestModel = EnsembleEvaluation(20)
        # randomForestModel.initDataset(dataset, self.kFoldPartitions, digitsAttributeTypes)
        # randomForestAccuracy, randomForestF1 = randomForestModel.evaluate()

        for testIdx in range(10):
            trainingData = self.kFoldPartitions[:testIdx] + self.kFoldPartitions[testIdx+1:]
            trainingData = [instance for fold in trainingData for instance in fold]
            
            testData = self.kFoldPartitions[testIdx]

            featureBounds = self.getBounds(trainingData, testData)
            normalizedTrainingData = self._normalize(trainingData, featureBounds)
            normalizedTestData = self._normalize(testData, featureBounds)

            KNN_Accuracy, KNN_F1_Score = (self.evaluateKNN(normalizedTrainingData, normalizedTestData, datasetLabels))
            KNN_Accuracies.append(KNN_Accuracy)
            KNN_F1_Scores.append(KNN_F1_Score)

            # DecisionTreeAccuracy.append(self.getDecisionTreeAccuracy(normalizedTrainingData, normalizedTestData))
            # print(self.getDecisionTreeAccuracy(trainingData, testData))
        
        print(KNN_Accuracies)
        print(KNN_F1_Scores)
        
        # print(DecisionTreeAccuracy)
    
    def getDecisionTreeAccuracy(self, trainingData, testData):
        decisionTree = DecisionTree(trainingData, testData)
        decisionTree.train()
        return decisionTree.trainingAccuracy()
    
    def evaluateKNN(self, trainingData, testData, datasetLabels):
        KNNModel = KNN(10, trainingData, testData)
        return KNNModel.evaluate(datasetLabels)

    def getBounds(self, trainingData, testData):
        featureBounds = [[float('inf'), -float('inf')] for _ in range(len(trainingData[0])-1)]

        for trainingInstance in trainingData:
            for featureIdx, featureVal in enumerate(trainingInstance[:-1]):
                featureBounds[featureIdx] = self.addToBounds(featureBounds[featureIdx], featureVal)
            
        return featureBounds
    
    def addToBounds(self, oldBound, newValue):
        """
        Incorporates a new instance to the boundary of a feauture
        """
        minOldBound, maxOldBound = oldBound
        return [min(minOldBound, newValue), max(maxOldBound, newValue)]

    def _normalize(self, dataset, featureBounds):
        """
        Normalizes all features of the array between 0 and 1

        returns: None
        """
        normalizedData = []
        
        for instance in dataset:
            normalizedInstance = []
            for featureIdx, featureValue in enumerate(instance):

                if featureIdx == len(instance)-1:
                    normalizedInstance.append(featureValue)

                else:
                    if featureBounds[featureIdx][1] - featureBounds[featureIdx][0] == 0:
                        normalizedValue = featureValue
                    else:
                        normalizedValue = (featureValue - featureBounds[featureIdx][0])/(featureBounds[featureIdx][1] - featureBounds[featureIdx][0])
                    normalizedInstance.append(normalizedValue)

            normalizedData.append(normalizedInstance)
        
        return normalizedData
        

digitsModel = RunModels()
digitsModel.runDigitsModels()
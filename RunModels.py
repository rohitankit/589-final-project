from LoadDataset import LoadDataset
from Graph import Graph

from KNN.KNN import KNN
from DecisionTree.DecisionTree import DecisionTree
from RandomForest.Main import EnsembleEvaluation


from statistics import mean

class RunModels:

    def __init__(self):
        self.kFoldPartitions = []
        self.classIdx = 0

    def runDigitsModels(self):
        digitsData = LoadDataset()
        dataset, self.kFoldPartitions = digitsData.loadDigitDataset()

        # Attribute types for Random Forest - False for numerical and True for Categorical
        digitsAttributeTypes = [False]*(len(dataset[0])-1) + [True]
        self.classIdx = len(dataset[0])-1

        # dataset
        datasetLabels = set()
        for instance in dataset:
            datasetLabels.add(instance[self.classIdx])
        datasetLabels = list(datasetLabels)
        
        # self.getRandomForestTable(dataset, digitsAttributeTypes)
        # self.getKnnTable(datasetLabels)

        # self.getKnnGraph(datasetLabels)
        # self.getRandomForestGraph(dataset, digitsAttributeTypes)

    def runTitanicModels(self):
        titanicData = LoadDataset()
        ignoreAttributes = [2]
        dataset, self.kFoldPartitions = titanicData.loadTitanicDataset(ignoreAttributes)

        self.classIdx = 0
        titanicAttributeTypes = [True, True, True, False, False, False, False]

        datasetLabels = [0, 1]
        print("\n")
        self.getRandomForestTable(dataset, titanicAttributeTypes)
        print("\n")
        self.getKnnTable(datasetLabels)



    def getKnnGraph(self, datasetLabels):
        KNN_hyperparameters = [1, 3, 5, 7, 9, 11, 13, 15]
        KNN_K_Accuracies = []
        KNN_K_F1Score = []

        for k in KNN_hyperparameters:
            KNN_CrossValidation_Accuracies = []
            KNN_CrossValidation_F1_Scores = []
            for testIdx in range(10):
                trainingData = self.kFoldPartitions[:testIdx] + self.kFoldPartitions[testIdx+1:]
                trainingData = [instance for fold in trainingData for instance in fold]
                
                testData = self.kFoldPartitions[testIdx]

                featureBounds = self.getBounds(trainingData, testData)
                normalizedTrainingData = self._normalize(trainingData, featureBounds)
                normalizedTestData = self._normalize(testData, featureBounds)

                KNN_Accuracy, KNN_F1_Score = (self.evaluateKNN(k, normalizedTrainingData, normalizedTestData, datasetLabels, self.classIdx))
                KNN_CrossValidation_Accuracies.append(KNN_Accuracy)
                KNN_CrossValidation_F1_Scores.append(KNN_F1_Score)
            
            KNN_K_Accuracies.append(mean(KNN_CrossValidation_Accuracies))
            KNN_K_F1Score.append(mean(KNN_CrossValidation_F1_Scores))
        
        print(KNN_K_Accuracies)
        print(KNN_K_F1Score)
        KNN_Graph = Graph()
        KNN_Graph.setTitle("KNN - Accuracy and F1 scores vs K")
        KNN_Graph.setXLabel("K values")
        KNN_Graph.setYLabel("Accuracy and F1 scores")
        KNN_Graph.plot(KNN_hyperparameters, KNN_K_Accuracies, "Accuracy Scores")
        KNN_Graph.plot(KNN_hyperparameters, KNN_K_F1Score, "F1 Scores")
        KNN_Graph.show()

    def getRandomForestGraph(self, dataset, digitsAttributeTypes):
        randomForestHyperParameters = [5, 10, 15, 20, 25, 30]
        randomForestAccuracies = []
        randomForestF1Scores = []

        for k in randomForestHyperParameters:
            randomForestModel = EnsembleEvaluation(k)
            randomForestModel.initDataset(dataset, self.kFoldPartitions, digitsAttributeTypes, self.classIdx)
            randomForestAccuracy, randomForestF1 = randomForestModel.evaluate()
            randomForestAccuracies.append(randomForestAccuracy)
            randomForestF1Scores.append(randomForestF1)
        
        print(randomForestAccuracies)
        print(randomForestF1Scores)
        KNN_Graph = Graph()
        KNN_Graph.setTitle("Random Forest Graph - Accuracy and F1 scores vs K")
        KNN_Graph.setXLabel("K values")
        KNN_Graph.setYLabel("Accuracy and F1 scores")
        KNN_Graph.plot(randomForestHyperParameters, randomForestAccuracies, "Accuracy Scores")
        KNN_Graph.plot(randomForestHyperParameters, randomForestF1Scores, "F1 Scores")
        KNN_Graph.show()
        

    def getRandomForestTable(self, dataset, digitsAttributeTypes):
        randomForestHyperParameters = [5, 10, 15, 20]

        print("Dataset 2 metric evaluation using random forests\n")
        print("  k-value  |  Accuracy  | F1 Score ")
        print("------------------------------------")

        for k in randomForestHyperParameters:
            randomForestModel = EnsembleEvaluation(k)
            randomForestModel.initDataset(dataset, self.kFoldPartitions, digitsAttributeTypes, self.classIdx)
            randomForestAccuracy, randomForestF1 = randomForestModel.evaluate()

            print("  {:.4f}  |   {:.4f}   |  {:.4f}".format(k, randomForestAccuracy, randomForestF1))
    
    def getKnnTable(self, datasetLabels):
        KNN_hyperparameters = [3, 5, 7, 9, 11]
        KNN_Accuracies = []
        KNN_F1_Scores = []

        print("Dataset 2 metric evaluation using KNN\n")
        print("  k-value  |  Accuracy  | F1 Score ")
        print("------------------------------------")

        for k in KNN_hyperparameters:
            for testIdx in range(10):
                trainingData = self.kFoldPartitions[:testIdx] + self.kFoldPartitions[testIdx+1:]
                trainingData = [instance for fold in trainingData for instance in fold]
                
                testData = self.kFoldPartitions[testIdx]

                featureBounds = self.getBounds(trainingData, testData)
                normalizedTrainingData = self._normalize(trainingData, featureBounds)
                normalizedTestData = self._normalize(testData, featureBounds)

                KNN_Accuracy, KNN_F1_Score = (self.evaluateKNN(k, normalizedTrainingData, normalizedTestData, datasetLabels, self.classIdx))
                KNN_Accuracies.append(KNN_Accuracy)
                KNN_F1_Scores.append(KNN_F1_Score)
            
            print("  {:.4f}  |   {:.4f}   |  {:.4f}".format(k, mean(KNN_Accuracies), mean(KNN_F1_Scores)))
    
    def evaluateDecisionTree(self, trainingData, testData):
        decisionTree = DecisionTree(trainingData, testData)
        decisionTree.train()
        return decisionTree.trainingAccuracy()
    
    def evaluateKNN(self, k, trainingData, testData, datasetLabels, classIdx):
        KNNModel = KNN(k, trainingData, testData)
        return KNNModel.evaluate(datasetLabels, classIdx)

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
digitsModel.runTitanicModels()
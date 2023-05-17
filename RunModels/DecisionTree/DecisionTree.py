from .TreeCreator import TreeCreator

class DecisionTree:

    def __init__(self, trainingData, testData):
        self.rootNode = None
        self.trainingData = trainingData
        self.testData = testData
    
    def loadData(self, trainingData, testData):
        self.trainingData = trainingData
        self.testData = testData
    
    def train(self):
        features = list(range(len(self.trainingData[0])-1))
        tree = TreeCreator(features, self.trainingData)
        self.rootNode = tree.create()
    
    def testAccuracy(self):
        correctPredictions = 0
        for testInstance in self.testData:
            predictedLabel = self.rootNode.predict(testInstance)
            if predictedLabel == testInstance[-1]:
                correctPredictions += 1
            # else:
                # print(" ")
                # predictedLabel = self.rootNode.predict(testInstance)
                # print(predictedLabel, testInstance[-1])
        
        return correctPredictions/len(self.testData)
    
    def trainingAccuracy(self):
        correctPredictions = 0
        for trainingInstance in self.trainingData:
            predictedLabel = self.rootNode.predict(trainingInstance)
            if predictedLabel == trainingInstance[-1]:
                correctPredictions += 1
            # else:
            #     print(predictedLabel, trainingInstance[-1])
        
        return correctPredictions/len(self.trainingData)
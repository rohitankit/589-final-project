import statistics

from .RandomForest import RandomForest
from .ModelEvaluation import ModelEvaluation

class EnsembleEvaluation:

    NUMBER_OF_FOLDS = 10

    def __init__(self, k):
        self.randomForest = RandomForest(k)

        self.dataset = None
        self.datasetLabelIdx = -1
        self.attributeTypes = []
        self.kFoldDataset = []
    
    def initDataset(self, dataset, kFoldDataset, attributeTypes):
        self.dataset = dataset
        self.kFoldDataset = kFoldDataset

        self.datasetLabelIdx = len(self.dataset[0])-1
        self.attributeTypes = attributeTypes

    def getLabels(self):
        labelSet = set()
        for instance in self.dataset:
            label = instance[self.datasetLabelIdx]
            if label not in labelSet:
                labelSet.add(label)
        
        return labelSet
    
    def evaluate(self):
        # metricsAverage = [0, 0, 0, 0]
        accuracyValues = []
        F1_Values = []

        for testIdx in range(EnsembleEvaluation.NUMBER_OF_FOLDS):
            modelEvaluator = ModelEvaluation()
            modelEvaluator.setLabels(self.getLabels())

            trainingData = self.kFoldDataset[:testIdx] + self.kFoldDataset[testIdx+1:]
            trainingData = [instance for fold in trainingData for instance in fold]
            
            testData = self.kFoldDataset[testIdx]

            self.train(trainingData)

            for testInstance in testData:
                predictedLabel = self.randomForest.predict(testInstance)
                actualLabel = testInstance[-1]
                modelEvaluator.addInstanceEvaluation(int(actualLabel), int(predictedLabel))
            
            accuracyValues.append(modelEvaluator.getAccuracy())
            F1_Values.append(modelEvaluator.getF1Score())
        
        return statistics.mean(accuracyValues), statistics.mean(F1_Values)
            
    def train(self, trainingData):
        self.randomForest.train(trainingData, self.attributeTypes, self.datasetLabelIdx)           

    def test(self, testData, modelEvaluator):
        for testInstance in testData:
            modelEvaluator.addInstanceEvaluation(int(testInstance[self.datasetLabelIdx]), int(self.randomForest.predict(testInstance)))
        
        metricsArray = [modelEvaluator.getAccuracy(),
                        modelEvaluator.getPrecision(),
                        modelEvaluator.getRecall(),
                        modelEvaluator.getF1Score()]
        
        return metricsArray

'''
def ensembleGraphEvaluation(datasetName, datasetPath, datasetDelimiter, datasetLabelIdx, attributeTypes):

    nTreeValues = [1,5,10,20,30,40,50, 100, 200, 300]

    graphMetrics = {"Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "F1": []}
    
    for k in nTreeValues:
        ensembleEval = EnsembleEvaluation(k)
        ensembleEval.initDataset(datasetPath, datasetDelimiter, datasetLabelIdx, attributeTypes)
        ensembleMetrics = ensembleEval.evaluate()

        graphMetrics["Accuracy"].append(ensembleMetrics[0])
        graphMetrics["Precision"].append(ensembleMetrics[1])
        graphMetrics["Recall"].append(ensembleMetrics[2])
        graphMetrics["F1"].append(ensembleMetrics[3])

    accuracyGraph = Graph()
    accuracyGraph.setTitle(f"Accuracy vs nTree graph for {datasetName} dataset")
    accuracyGraph.setXLabel("nTree Values")
    accuracyGraph.setXValues(nTreeValues)
    accuracyGraph.setYLabel("Accuracy Values")
    accuracyGraph.setYValues(graphMetrics["Accuracy"])

    precisionGraph = Graph()
    precisionGraph.setTitle(f"Precision vs nTree graph for {datasetName} dataset")
    precisionGraph.setXLabel("nTree Values")
    precisionGraph.setXValues(nTreeValues)
    precisionGraph.setYLabel("Precision Values")
    precisionGraph.setYValues(graphMetrics["Precision"])

    recallGraph = Graph()
    recallGraph.setTitle(f"Recall vs nTree graph for {datasetName} dataset")
    recallGraph.setXLabel("nTree Values")
    recallGraph.setXValues(nTreeValues)
    recallGraph.setYLabel("Recall Values")
    recallGraph.setYValues(graphMetrics["Recall"])

    F1Graph = Graph()
    F1Graph.setTitle(f"F1 vs nTree graph for {datasetName} dataset")
    F1Graph.setXLabel("nTree Values")
    F1Graph.setXValues(nTreeValues)
    F1Graph.setYLabel("F1 Values")
    F1Graph.setYValues(graphMetrics["F1"])
    
    accuracyGraph.show()
    precisionGraph.show()
    recallGraph.show()
    F1Graph.show()

def main():
    WINE_DATASET_PATH = './assets/hw3_cancer.csv'
    WINE_DATASET_DELIMITER = '\t'
    WINE_DATASET_CLASS_IDX = 9
    wineAttributeTypes = [False]*9 + [True]

    ensembleGraphEvaluation("Cancer",
                            WINE_DATASET_PATH,
                            WINE_DATASET_DELIMITER,
                            WINE_DATASET_CLASS_IDX,
                            wineAttributeTypes)

    # CONGRESS_DATASET_PATH = './assets/hw3_house_votes_84.csv'
    # CONGRESS_DATASET_DELIMITER = ','
    # CONGRESS_DATASET_CLASS_IDX = 16
    # congressAttributeTypes = [True] * 17

    # ensembleGraphEvaluation("House Votes",
    #                         CONGRESS_DATASET_PATH,
    #                         CONGRESS_DATASET_DELIMITER,
    #                         CONGRESS_DATASET_CLASS_IDX,
    #                         congressAttributeTypes)

main()
'''
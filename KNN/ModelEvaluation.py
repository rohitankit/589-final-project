from statistics import mean

class ModelEvaluation:

    POSITIVE_LABEL = 0
    NEGATIVE_LABEL = 1

    def __init__(self):
        self.evaluationMatrix = []
        self.evaluationDimensions = 0
        self.totalClassifications = 0
        self.labelOffset = 0
    
    def setLabels(self, labelSet):
        self.labelOffset = int(min(labelSet))
        self.evaluationMatrix = [[0]*len(labelSet) for _ in range(len(labelSet))]
        self.evaluationDimensions = len(labelSet)
    
    def addInstanceEvaluation(self, trueClass, predictedClass):
        self.totalClassifications += 1
        self.evaluationMatrix[int(trueClass)-int(self.labelOffset)][predictedClass-self.labelOffset] += 1
    
    def getConfusionMatrix(self):
        for arr in self.evaluationMatrix:
            print(arr)
        print("\n")

    def getAccuracy(self):
        classifiedCorrectly = 0
        for i in range(self.evaluationDimensions):
            classifiedCorrectly += self.evaluationMatrix[i][i]
        
        return classifiedCorrectly/(self.totalClassifications)
    
    def getPrecision(self):
        return mean(self.getLabelPrecisions())

    def getRecall(self):
        return mean(self.getLabelRecalls())

    def getLabelPrecisions(self):
        labelPrecisions = []
        for predictedClassIdx in range(self.evaluationDimensions):
            truePositives = self.evaluationMatrix[predictedClassIdx][predictedClassIdx]
            falsePositives = 0
            for trueClassIdx in range(self.evaluationDimensions):
                if predictedClassIdx != trueClassIdx:
                    falsePositives += self.evaluationMatrix[trueClassIdx][predictedClassIdx]
            
            labelPrecisions.append(truePositives/(truePositives+falsePositives))
        
        return labelPrecisions

    def getLabelRecalls(self):
        labelRecalls = []
        for trueClassIdx in range(self.evaluationDimensions):
            truePositives = self.evaluationMatrix[trueClassIdx][trueClassIdx]
            falseNegatives = 0
            for predictedClassIdx in range(self.evaluationDimensions):
                if predictedClassIdx != trueClassIdx:
                    falseNegatives += self.evaluationMatrix[trueClassIdx][predictedClassIdx]
            
            labelRecalls.append(truePositives/(truePositives+falseNegatives))

        return labelRecalls
    
    def getF1Score(self):
        labelPrecisions = self.getLabelPrecisions()
        labelRecalls = self.getLabelRecalls()
        F1Total = 0

        for i in range(len(labelPrecisions)):
            F1Total += ModelEvaluation.getHarmonicAverage(labelRecalls[i], labelPrecisions[i])
        
        return F1Total/len(labelPrecisions)

    def getHarmonicAverage(recall, precision):
        if recall == 0 or precision == 0:
            return 0
        return 2 / ((1/recall) + (1/precision))
        
        
from run import load_training_set, load_test_set
from collections import Counter
from math import e as e
from math import log as log
import matplotlib.pyplot as plt

class BayesModel:

    POSITIVE_LABEL = 0
    NEGATIVE_LABEL = 1

    def __init__(self):
        
        self.labelWordCount = [{}, {}]
        self.labelTrainingSize = [0, 0]
        self.labelDictionarySize = 0
        self.positiveLabelProbabaility = 0
        self.negativeLabelProbabaility = 0
    
    def train(self, positiveTrainingSize, negativeTrainingSize):

        positiveInstances, negativeInstances, vocab = load_training_set(positiveTrainingSize, negativeTrainingSize)

        self.positiveLabelProbabaility = len(positiveInstances) / (len(positiveInstances) + len(negativeInstances))
        self.negativeLabelProbabaility = 1 - self.positiveLabelProbabaility

        self.labelWordCount[BayesModel.POSITIVE_LABEL], self.labelTrainingSize[BayesModel.POSITIVE_LABEL] = self.loadWordCount(positiveInstances)
        self.labelWordCount[BayesModel.NEGATIVE_LABEL], self.labelTrainingSize[BayesModel.NEGATIVE_LABEL] = self.loadWordCount(negativeInstances)

        self.labelDictionarySize = len(self.labelWordCount[BayesModel.POSITIVE_LABEL].keys())
        self.labelDictionarySize += len(self.labelWordCount[BayesModel.NEGATIVE_LABEL].keys())

    def loadWordCount(self, trainingInstances):

        dictionaryWordCount = {}
        totalWordCount = 0

        for instance in trainingInstances:
            for word in instance:
                dictionaryWordCount[word] = dictionaryWordCount.get(word, 0) + 1
                totalWordCount += 1
        
        return dictionaryWordCount, totalWordCount
    

    def test(self, positiveTestSize, negativeTestSize, laplaceCoefficient):

        positiveInstances, negativeInstances = load_test_set(positiveTestSize, negativeTestSize)

        # print("Original Naive Bayes Evaluation")
        # self.testHelper(positiveInstances, negativeInstances, laplaceCoefficient, self.predict)

        print(f"\nNaive Bayes using log Evaluation with alpha = {laplaceCoefficient}")
        return self.testHelper(positiveInstances, negativeInstances, laplaceCoefficient, self.predictLog)

    
    def testHelper(self, positiveInstances, negativeInstances, laplaceCoefficient, predictionFunction):
        BayesModelEvaluation = ModelEvaluation()

        for instance in positiveInstances:
            trueClass = BayesModel.POSITIVE_LABEL
            uniqueWords = set(instance)
            predictedClass = predictionFunction(uniqueWords, laplaceCoefficient)
            BayesModelEvaluation.addInstanceEvaluation(trueClass, predictedClass)
        
        for instance in negativeInstances:
            trueClass = BayesModel.NEGATIVE_LABEL
            uniqueWords = set(instance)
            predictedClass = predictionFunction(uniqueWords, laplaceCoefficient)
            BayesModelEvaluation.addInstanceEvaluation(trueClass, predictedClass)
        
        BayesModelEvaluation.getConfusionMatrix()
        print(f"Accuracy = {BayesModelEvaluation.getAccuracy()}")
        print(f"Precision = {BayesModelEvaluation.getPrecision()}")
        print(f"Recall = {BayesModelEvaluation.getRecall()}")
        print(f"Error Rate = {BayesModelEvaluation.getErrorRate()}")

        return BayesModelEvaluation.getAccuracy()

    def predict(self, instance, laplaceCoefficient):

        positivePredictionProb = self.positiveLabelProbabaility
        negativePredictionProb = self.negativeLabelProbabaility

        for word in instance:
            positivePredictionProb *= self.otherConditionalLabelProbability(BayesModel.POSITIVE_LABEL, word, laplaceCoefficient)
        
        for word in instance:
            negativePredictionProb *= self.otherConditionalLabelProbability(BayesModel.NEGATIVE_LABEL, word, laplaceCoefficient)

        if positivePredictionProb > negativePredictionProb:
            return BayesModel.POSITIVE_LABEL
        else:
            return BayesModel.NEGATIVE_LABEL
    
    def predictLog(self, instance, laplaceCoefficient):

        positivePredictionProb = self.positiveLabelProbabaility
        negativePredictionProb = self.negativeLabelProbabaility

        for word in instance:
            positivePredictionProb += log(self.conditionalLabelProbability(BayesModel.POSITIVE_LABEL, word, laplaceCoefficient))
            negativePredictionProb += log(self.conditionalLabelProbability(BayesModel.NEGATIVE_LABEL, word, laplaceCoefficient))
        
        if positivePredictionProb > negativePredictionProb:
            return BayesModel.POSITIVE_LABEL
        else:
            return BayesModel.NEGATIVE_LABEL
        
    def otherConditionalLabelProbability(self, label, word, laplaceCoefficient):

        labelWordFrequency = self.labelWordCount[label].get(word, 0) + laplaceCoefficient
        labelTotalWordCount = self.labelTrainingSize[label] + (laplaceCoefficient * self.labelDictionarySize)

        if self.labelWordCount[label].get(word, 0) == 0:
            return 1

        return labelWordFrequency/labelTotalWordCount
        
    def conditionalLabelProbability(self, label, word, laplaceCoefficient):

        labelWordFrequency = self.labelWordCount[label].get(word, 0) + laplaceCoefficient
        labelTotalWordCount = self.labelTrainingSize[label] + (laplaceCoefficient * self.labelDictionarySize)

        return labelWordFrequency/labelTotalWordCount

class ModelEvaluation:

    POSITIVE_LABEL = 0
    NEGATIVE_LABEL = 1

    def __init__(self):
        self.truePositive = 0
        self.falseNegative = 0
        self.falsePositive = 0
        self.trueNegative = 0
        self.totalClassifications = 0
    
    def addInstanceEvaluation(self, trueClass, predictedClass):
        self.totalClassifications += 1

        if trueClass == ModelEvaluation.POSITIVE_LABEL and predictedClass == ModelEvaluation.POSITIVE_LABEL:
            self.truePositive += 1  
        elif trueClass == ModelEvaluation.POSITIVE_LABEL and predictedClass == ModelEvaluation.NEGATIVE_LABEL:
            self.falseNegative += 1
        if trueClass == ModelEvaluation.NEGATIVE_LABEL and predictedClass == ModelEvaluation.POSITIVE_LABEL:
            self.falsePositive += 1
        if trueClass == ModelEvaluation.NEGATIVE_LABEL and predictedClass == ModelEvaluation.NEGATIVE_LABEL:
            self.trueNegative += 1
    
    def getConfusionMatrix(self):

        print("")
        print(f"              predicted class ")
        print("")
        print("             |    +    |    -   ")
        print(f" True     +  |  {self.truePositive:5d}  |  {self.falseNegative:5d}  ")
        print(f" Class    -  |  {self.falsePositive:5d}  |  {self.trueNegative:5d}  ")
        print("")

    def getAccuracy(self):
        return (self.truePositive+self.trueNegative)/(self.totalClassifications)

    def getPrecision(self):
        return (self.truePositive)/(self.truePositive+self.falsePositive)
    
    def getRecall(self):
        return (self.truePositive)/(self.truePositive+self.falseNegative)
    
    def getErrorRate(self):
        return (self.falseNegative+self.falsePositive)/self.totalClassifications
        
if __name__ == "__main__":
    movieClassifier = BayesModel()
    movieClassifier.train(0.1, 0.5)
    movieClassifier.test(1.0, 1.0, 10)

    # alpha = 0.0001
    # xValues = []
    # yValues = []

    # while alpha <= 1000:
    #     xValue = alpha
    #     yValue = movieClassifier.test(0.20, 0.20, alpha)

    #     xValues.append(xValue)
    #     yValues.append(yValue)
    #     alpha *= 10
    
    # plt.plot(xValues, yValues)
    
    # plt.xlabel('Alpha')
    # plt.ylabel('Accuracy')
    
    # plt.title('Accuracy vs Alpha plot')
    # plt.xscale("log")

    # plt.show()


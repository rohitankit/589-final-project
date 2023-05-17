from .TreeCreator import TreeCreator

class DecisionTree:

    MAX_TREE_DEPTH = 5

    def __init__(self):
        self.rootNode = None

    def train(self, trainingData, attributeTypes, labelIdx):
        featuresIdxList = list(range(len(trainingData[0])))
        featuresIdxList.remove(labelIdx)

        tree = TreeCreator(DecisionTree.MAX_TREE_DEPTH)
        self.rootNode = tree.create(featuresIdxList, trainingData, attributeTypes, labelIdx)
    
    def predict(self, testInstance):
        return self.rootNode.predict(testInstance)
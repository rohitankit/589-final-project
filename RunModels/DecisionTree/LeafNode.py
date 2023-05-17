from collections import defaultdict

class LeafNode:

    def __init__(self, trainingInstances):
        self.trainingInstances = trainingInstances
        self.LABEL_IDX = len(trainingInstances[0])-1
    
    def predict(self, testInstance):
        return self.getMajorityLabel()
    
    def getMajorityLabel(self):
        labelFreq = defaultdict(lambda :0)
        for instance in self.trainingInstances:
            instanceLabel = instance[self.LABEL_IDX]
            labelFreq[instanceLabel] += 1
        
        maxFreq = -1
        majorityLabel = 0
        for label, freq in labelFreq.items():
            if freq > maxFreq:
                maxFreq = freq
                majorityLabel = label
        
        return majorityLabel
        
from sklearn import datasets
import numpy as np

from Dataset import Dataset

class LoadData:

    def __init__(self):
        self.datasetInstances = []
        self.datasetLabels = []
        self.datasetLength = 0

        self.dataset = None
    
    def loadDigitDataset(self):
        digitDataset = datasets.load_digits(return_X_y = True)
        self.datasetInstances = digitDataset[0]
        self.datasetLabels = digitDataset[1]

        data = []
        for idx, instance in enumerate(self.datasetInstances):
            data.append(np.append(instance, self.datasetLabels[idx]))

        self.dataset = Dataset()
        self.dataset.load(data)
        
        return data, self.dataset.getKFoldPartitions(10)

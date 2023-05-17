from sklearn import datasets
import numpy as np

from Dataset import Dataset

class LoadDataset:
    
    def loadDigitDataset(self):
        digitDataset = datasets.load_digits(return_X_y = True)
        datasetInstances = digitDataset[0]
        datasetLabels = digitDataset[1]

        datasetWithLabels = []
        for idx, instance in enumerate(datasetInstances):
            datasetWithLabels.append(np.append(instance, datasetLabels[idx]))

        digitDatasetObj = Dataset()
        digitDatasetObj.load(datasetWithLabels)
        
        return datasetWithLabels, digitDatasetObj.getKFoldPartitions(10)

    def loadTitanicDataset(self, ignoreAttributes):
        titanicDatasetObj = Dataset()
        titanicDatasetObj.loadFromFile('./assets/titanic.csv', ',', 0, ignoreAttributes)

        dataset = titanicDatasetObj.getRawDataset()

        return dataset, titanicDatasetObj.getKFoldPartitions(10)

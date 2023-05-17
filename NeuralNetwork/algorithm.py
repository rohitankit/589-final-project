from abc import ABC

from data import VectorDataset

class VectorAlgorithm(ABC):
    def __init__(self, data: VectorDataset, **kwargs):
        pass

    def train(self, training_data: VectorDataset, testing_data: VectorDataset):
        pass

    def predict(self, input_data):
        pass
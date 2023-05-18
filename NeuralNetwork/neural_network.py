from typing import List, Optional

from perceptron import Perceptron

from data import DataVectorizer

from .ModelEvaluation import ModelEvaluation


class NeuralNetwork:
    def __init__(
        self,
        training_data: List[List],
        test_data: List[List],
        class_index: int = -1,
        hidden_layer_sizes: Optional[List[int]] = None,
        batch_size: int = 32,
        epochs: int = 1000,
        learning_rate: float = 1e-2,
        **kwargs
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [16]
            
        self.vectorizer = DataVectorizer(training_data + test_data, class_index)

        self.training_data = training_data
        self.test_data = test_data
        self.vectorized_training_data = self.vectorizer.vectorize_dataset(training_data)
        self.vectorized_test_data = self.vectorizer.vectorize_dataset(training_data)

        self.net = Perceptron(
            sizes=[
                self.vectorized_training_data.attributes.shape[-1],
                *hidden_layer_sizes,
                self.vectorized_training_data.labels.shape[-1],
            ]
        )

        self.net.train(
            batch_size=batch_size,
            epochs=epochs,
            train_x=self.vectorized_training_data.attributes,
            train_y=self.vectorized_training_data.labels,
            eval_x=self.vectorized_test_data.attributes,
            eval_y=self.vectorized_test_data.labels,
            learning_rate=learning_rate,
            **kwargs,
        )

    def evaluate(self, datasetLabels, classIdx):
        modelEvaluator = ModelEvaluation()
        modelEvaluator.setLabels(datasetLabels)

        for testInstance in self.testData:
            predictedClass = self.predict(testInstance, classIdx)
            actualClass = testInstance[classIdx]

            modelEvaluator.addInstanceEvaluation(int(actualClass), int(predictedClass))

        return modelEvaluator.getAccuracy(), modelEvaluator.getF1Score()

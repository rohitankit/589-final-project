from typing import List, Optional

from .perceptron import Perceptron

from .data import DataVectorizer

from .ModelEvaluation import ModelEvaluation


class NeuralNetwork:
    def __init__(
        self,
        training_data: List[List],
        test_data: List[List],
        class_index: int = -1,
        hidden_layer_sizes: Optional[List[int]] = None,
        batch_size: int = 32,
        epochs: int = 500,
        learning_rate: float = 1.0,
        **kwargs
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [16]
            
        self.class_index = class_index
        self.vectorizer = DataVectorizer(training_data + test_data, self.class_index)
        
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
            console_log=False,
            show_progress_bar=False,
            **kwargs,
        )
        
    def predict(self, row: List, class_idx: int):
        
        x_data, _ = self.vectorizer.vectorize_datapoint(row)
        
        y_pred = self.net.forward(x_data)
        
        return self.vectorizer.vectorizers[self.class_index].devectorize(y_pred)
        

    def evaluate(self, datasetLabels, classIdx):
        modelEvaluator = ModelEvaluation()
        modelEvaluator.setLabels(datasetLabels)

        for testInstance in self.test_data:
            predictedClass = self.predict(testInstance, classIdx)
            actualClass = testInstance[classIdx]

            modelEvaluator.addInstanceEvaluation(int(actualClass), int(predictedClass))

        return modelEvaluator.getAccuracy(), modelEvaluator.getF1Score()

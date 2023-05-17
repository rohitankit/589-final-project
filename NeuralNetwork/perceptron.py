from typing import List, Tuple, Dict, Optional

import numpy as np

from fc_layer import FCLayer


def cost_function(ground_truth: np.ndarray, network_out: np.ndarray) -> np.ndarray:
    """Calculates the cost of the network output compared to the ground truth

    Args:
        ground_truth (np.ndarray): The ground truth labels
        network_out (np.ndarray): The output of the network

    Returns:
        np.ndarray: The cost
    """
    cost = -ground_truth * np.log(network_out) - (1 - ground_truth) * np.log(
        1 - network_out
    )
    cost = cost.squeeze()
    return cost


class Perceptron:
    """This class stores a stack of layers and allows us to train them using backpropagation"""

    def __init__(self, sizes: List[int], lamda: float = 0.0) -> None:
        """Initializes the perceptron

        Args:
            sizes (List[int]): Sizes for layers. First element should be input size, last should be output size
            lamda (float, optional): Regularization parameter. Defaults to 0.0
        """

        # fmt: off
        self.layers = [FCLayer(sizes[i], sizes[i + 1])
                       for i in range(len(sizes) - 1)]
        # fmt: on
        self.lamda = lamda

    def forward(self, input_x: np.ndarray) -> np.ndarray:
        """Does a forward pass on the network"""

        intermediate = input_x

        for layer in self.layers:
            intermediate = layer(intermediate)

        return intermediate

    def forward_backward(
        self, input_x: np.ndarray, input_y: np.ndarray
    ) -> Tuple[
        np.ndarray,
        List[Tuple[np.ndarray, np.ndarray]],
        np.ndarray,
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        """Computes the gradients of all weights using backpropagation

        Args:
            input_x (np.ndarray): Network input
            input_y (np.ndarray): Expected output of the network

        Returns:
            Tuple containing:
                np.ndarray: The output of the network
                List[Tuple[np.ndarray, np.ndarray]]: Gradients for each layer and example
                np.ndarray: Cost for each training example
                List[np.ndarray]: Activations for each layer and example
                List[np.ndarray]: Intermediate results (z-values) for each layer and example
                List[np.ndarray]: Delta values for each layer and example
        """

        activations: List[np.ndarray] = [input_x]
        intermediates: List[np.ndarray] = []

        # Sequentially apply all layers in forward pass
        for layer in self.layers:
            last_act = activations[-1]
            this_act, this_intermediate = layer.forward(last_act)
            activations.append(this_act)
            intermediates.append(this_intermediate)

        # Initialize gradient result buffers
        network_out = activations[-1]

        grads: List[Tuple[np.ndarray, np.ndarray]] = []
        deltas: List[np.ndarray] = [network_out - input_y]

        # Compare final layer output to ground truth to calculate cost
        cost = cost_function(input_y, network_out)

        # Do the backwards pass
        acts_and_layers = list(zip(activations, self.layers))
        for layer_acts, layer in reversed(acts_and_layers):
            # Calculate the new deltas for this layer
            next_delta, weight_grad, bias_grad = layer.backward(layer_acts, deltas[0])
            deltas.insert(0, next_delta)

            # Add regularization
            weight_grad += self.lamda * layer.weight_theta

            grads.insert(0, (weight_grad, bias_grad))

        # Return this massive tuple of everything (I kind of hate this)
        return network_out, grads, cost, activations, intermediates, deltas

    def get_thetas(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Gets the weights and biases of each layer in the perceptron"""

        thetas = [(layer.weight_theta, layer.bias_theta) for layer in self.layers]

        return thetas

    def set_thetas(self, new_thetas: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Sets the weights and biases of each layer in the perceptron"""

        for layer, new_theta in zip(self.layers, new_thetas):
            layer.theta.weight, layer.theta.bias = new_theta

    def sgd_step(self, train_x: np.ndarray, train_y: np.ndarray, learning_rate: float):
        """This function performs a single step of SGD on the perceptron

        Args:
            train_x (np.ndarray): The input data
            train_y (np.ndarray): The ground truth labels
            learning_rate (float): The learning rate

        Returns:
            float: The mean cost over the batch
        """

        _, each_layer_each_input_grads, cost, _, _, _ = self.forward_backward(
            train_x, train_y
        )

        # Accumulate cost over the batch dimension
        cost = cost.mean(axis=0)

        # Iterate over each layer and update their weights and biases
        for layer, each_input_grad in zip(self.layers, each_layer_each_input_grads):
            # Accumulate gradients over the batch dimension
            weight_grad = each_input_grad[0].mean(axis=0)
            bias_grad = each_input_grad[1].mean(axis=0)

            layer.theta.update(
                weight_grad=weight_grad,
                bias_grad=bias_grad,
                learning_rate=learning_rate,
            )

        return cost

    def evaluate(self, eval_x: np.ndarray, eval_y: np.ndarray) -> Dict[str, float]:
        """This function evaluates the perceptron on the given data

        Args:
            eval_x (np.ndarray): The input data
            eval_y (np.ndarray): The ground truth labels

        Returns:
            Dict[str, float]: A dictionary of metrics
        """

        network_out = self.forward(eval_x)
        cost = cost_function(eval_y, network_out)

        corrects = np.argmax(network_out, axis=1) == np.argmax(eval_y, axis=1)
        accuracy = corrects.mean()

        return {"accuracy": accuracy, "cost": cost.mean()}

    def batch_iterator(self, data_x: np.ndarray, data_y: np.ndarray, batch_size: int):
        """

        Args:
            input_x (np.ndarray): Input data
            input_y (np.ndarray): Input ground truth
            batch_size (int): batch size

        Yields:
            Tuple[np.ndarray, np.ndarray]: Tuples of (input, ground truth) for each batch
        """

        dataset_size = data_x.shape[0]

        epoch_index_order = np.arange(dataset_size)
        np.random.shuffle(epoch_index_order)

        num_batches = dataset_size // batch_size

        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = batch_start + batch_size
            batch_indices = epoch_index_order[batch_start:batch_end]
            yield data_x[batch_indices], data_y[batch_indices]

    def train(
        self,
        batch_size: int,
        epochs: int,
        train_x: np.ndarray,
        train_y: np.ndarray,
        eval_x: np.ndarray,
        eval_y: np.ndarray,
        learning_rate: float = 1e-2,
        console_log=True,
        show_progress_bar: bool = False,
        train_name: Optional[str] = None,
    ) -> Dict[int, float]:
        """_summary_

        Args:
            batch_size (int): Size of batches to split training data for each SGD step
            epochs (int): Number of passes over training data
            train_x (np.ndarray): Input training data
            train_y (np.ndarray): Ground truth labels for trianing data
            eval_x (np.ndarray): Input test data
            eval_y (np.ndarray): Ground truth labels for testing data
            learning_rate (float, optional): Size of step for SGD. Defaults to 1e-2
            console_log (bool, optional): Whether to print debug logs to console. Defaults to True
            show_progress_bar (bool, optional): Whether to show TQDM bar. Defaults to False
            train_name (Optional[str], optional): Name for TQDM bar. Defaults to None

        Returns:
            Dict[int, float]: A dictionary of number of training examples shown and the cost
        """

        perfs: Dict[int, float] = {}

        epoch_iterator = range(epochs)
        if show_progress_bar:
            try:
                from tqdm import tqdm  # pylint: disable=import-outside-toplevel

                epoch_iterator = tqdm(epoch_iterator, desc=train_name, leave=False)
            except ImportError:
                pass

        for epoch in epoch_iterator:
            epoch_train_costs = []

            batch_iterator = self.batch_iterator(train_x, train_y, batch_size)

            # Iterate for each batch of data in the epoch
            for batch_train_x, batch_train_y in batch_iterator:
                cost = self.sgd_step(batch_train_x, batch_train_y, learning_rate)

                epoch_train_costs.append(cost)

            # Evaluate after epoch
            train_eval = self.evaluate(train_x, train_y)
            test_eval = self.evaluate(eval_x, eval_y)

            number_of_examples_seen = (epoch + 1) * train_x.shape[0]
            perfs[number_of_examples_seen] = test_eval["cost"]

            if console_log:
                print(
                    f"Epoch {epoch} completed.\n\tEval cost: {test_eval['cost']}\n\tTrain cost: {train_eval['cost']}\n\tEval accuracy: {test_eval['accuracy']}\n\tTrain accuracy: {train_eval['accuracy']}"
                )

        return perfs

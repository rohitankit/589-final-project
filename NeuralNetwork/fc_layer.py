"""
This module defines the FCLayer class and the FCLayerTheta dataclass. The FCLayer class is an
implementation of a fully connected layer of a neural network, and the FCLayerTheta dataclass
represents the weights and biases of a fully connected layer.
"""

from typing import Tuple, Callable, Optional
from dataclasses import dataclass

import numpy as np


def sigmoid(input_x: np.ndarray) -> np.ndarray:
    """This function just calculates the sigmoid of the input of arbitrary dimension using numpy"""
    return 1 / (1 + np.exp(-input_x))


@dataclass
class FCLayerTheta:
    """This Dataclass stores the bias and weight parameters for a Fully Connected Layer """

    bias: np.ndarray
    weight: np.ndarray

    @staticmethod
    def random(in_dim: int, out_dim: int) -> "FCLayerTheta":
        """This function returns a randomly initialized FCLayerTheta object sampled from a gaussian
        distribution

        Args:
            in_dim (int): Input dimension for the layer
            out_dim (int): Output dimension for the layer

        Returns:
            FCLayerTheta: A randomly initialized set of weights and biases for a Fully Connected
            Layer
        """

        random_bias = np.random.randn(out_dim)
        random_weight = np.random.randn(out_dim, in_dim)

        return FCLayerTheta(
            bias=random_bias,
            weight=random_weight,
        )

    def update(
        self, weight_grad: np.ndarray, bias_grad: np.ndarray, learning_rate: float
    ) -> None:
        """This function updates the weights and biases of the layer

        Args:
            weight_grad (np.ndarray): gradient of the weights
            bias_grad (np.ndarray): gradient of the biases
            learning_rate (float): learning rate
        """

        self.weight -= learning_rate * weight_grad
        self.bias -= learning_rate * bias_grad


# @dataclass
# class FCLayerTheta


class FCLayer:
    """A Fully Connected Layer object"""

    def __init__(
        self,
        theta: FCLayerTheta,
        act: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.theta = theta

        if act is None:
            self.act = sigmoid
        else:
            self.act = act

    @staticmethod
    def random(
        in_dim: int, out_dim: int, act: Optional[Callable[[np.ndarray], np.ndarray]]
    ) -> "FCLayer":
        """This function returns a randomly initialized FCLayer object

        Args:
            in_dim (int): Input dimension for the layer
            out_dim (int): Output dimension for the layer
            act (Optional[Callable[[np.ndarray], np.ndarray]]): This is the activation function for
            the layer. Defaults to sigmoid.

        Returns:
            FCLayer: A randomly initialized FCLayer object
        """

        random_theta = FCLayerTheta.random(in_dim, out_dim)

        return FCLayer(
            theta=random_theta,
            act=act,
        )

    def __call__(self, layer_input: np.ndarray):
        act, _ = self.forward(layer_input)
        return act

    def forward(self, input_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Do the forward pass. Return activations and z values

        Args:
            input_x (np.ndarray): this layer's inputs

        Returns:
            Tuple[np.ndarray, np.ndarray]: (activations, z values) where z values are the inputs to
            the activation function
        """

        # Matrix multiplication over arbitrary shape batches
        intermediates = (
            np.einsum("ij, ...j -> ...i", self.theta.weight, input_x) + self.theta.bias
        )
        activations = self.act(intermediates)

        return activations, intermediates

    def backward(
        self, acts: np.ndarray, delta_next: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Do the backward pass. Return deltas and grad

        Args:
            acts (np.ndarray): this layer's inputs
            delta_next (np.ndarray): next layers delta values

        Returns:
            Tuple[np.ndarray, np.ndarray]: (delta, weight_grad, bias_grad)
        """

        # Transpose multiply by next layer deltas, then inner product by a and 1-a
        delta = np.einsum(
            "ji, ...j, ...i, ...i -> ...i",
            self.theta.weight,
            delta_next,
            acts,
            1 - acts,
        )

        # Outer product of activations and next layer deltas for gradient
        weight_grad = np.einsum(
            "...i, ...j -> ...ji",
            acts,
            delta_next,
        )
        bias_grad = delta_next.copy()

        return delta, weight_grad, bias_grad

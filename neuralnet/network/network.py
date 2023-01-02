import numpy as np
import numpy.typing as npt
from typing import Callable, Self, Sequence, Optional


class Layer:
    def __init__(self, inputlen: int, neurons: int, weightinit: tuple[float, float] = (-1.0, 1.0), biasinit: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.input = np.zeros(inputlen)
        self.weights = np.random.uniform(weightinit[0], weightinit[1], size=(neurons, inputlen)).round(1)
        self.biases = np.random.uniform(biasinit[0], biasinit[1], neurons).round(1)
        self.output = np.zeros(neurons)

    def feed_input(self, input: Sequence[float] | npt.NDArray[np.float64] | Self) -> npt.NDArray[np.float64]:
        if isinstance(input, Sequence) or isinstance(input, np.ndarray):
            input_vector = np.array(input)
        elif isinstance(input, Layer):
            input_vector = input.output
        else:
            raise TypeError(f"Wrong input type: {type(input)}")
        if len(input_vector) != len(self.input):
            raise ValueError(f"Bad input shape. Expected {len(self.input)}, got {len(input_vector)}")
        self.input = input_vector
        return self.input

    def calculate_output(self, activation_func: Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None) -> npt.NDArray[np.float64]:
        result = np.dot(self.weights, self.input) + self.biases
        if activation_func:
            result = activation_func(result)
        self.output = result
        return result


class NeuralNetwork:
    def __init__(self, inputs: int) -> None:
        self.input = np.zeros(inputs)
        self.layers: list[Layer] = []

    def feed_input(self, input: Sequence[float] | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not isinstance(input, Sequence) and not isinstance(input, np.ndarray):
            raise TypeError(f"Expected array-like input, but got {type(input)}")
        if len(input) != len(self.input):
            raise ValueError(f"Bad input shape. Expected {len(self.input)}, got {len(input)}")
        input_vector = np.array(input)
        self.input = input_vector
        return input_vector

    def add_layer(self, nodes: int, weightinit: tuple[float, float] = (-1.0, 1.0), biasinit: tuple[float, float] = (-1.0, 1.0)) -> Layer:
        if not self.layers:
            inputs = len(self.input)
        else:
            inputs = len(self.layers[-1].output)
        new_layer = Layer(inputs, nodes, weightinit, biasinit)
        self.layers.append(new_layer)
        return new_layer

    def forward_pass(self, activation_func: Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None) -> npt.NDArray[np.float64]:
        for i, layer in enumerate(self.layers):
            if i == 0:
                inputs = self.input
            else:
                inputs = self.layers[i-1].output
            layer.feed_input(inputs)
            layer.calculate_output(activation_func)
        return self.layers[-1].output

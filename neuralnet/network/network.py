import numpy as np
import numpy.typing as npt
from typing import Callable, Self, Sequence, Optional


class NeuralNetwork:
    def __init__(self) -> None:
        raise NotImplementedError


class Layer:
    def __init__(self, inputlen: int, neurons: int, weightinit: tuple[float, float] = (-1.0, 1.0), biasinit: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.input = np.zeros(inputlen)
        self.weights = np.random.uniform(weightinit[0], weightinit[1], size=(neurons, inputlen))
        self.biases = np.random.uniform(biasinit[0], biasinit[1], neurons)
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

    def calculate_output(self, activation_func: Optional[Callable] = None) -> npt.NDArray[np.float64]:
        raise NotImplementedError

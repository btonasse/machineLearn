import numpy as np
import numpy.typing as npt
from typing import Callable, Self, Sequence, Optional
import logging
from utils.logger import log_exceptions


class Layer:
    """
    Implementation of an individual layer of a neural network

    Initialization args:
        inputlen (int): the number of inputs the layer takes
        neurons (int): the number of nodes of the layer
        weightinit (tuple[float, float], optional): The range within which to initialize the layer's weights. Defaults to (-1.0, 1.0).
        biasinit (tuple[float, float], optional): The range within which to initialize the layer's biases. Defaults to (-1.0, 1.0).
    """

    def __init__(self, inputlen: int, neurons: int, weightinit: tuple[float, float] = (-1.0, 1.0), biasinit: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.input = np.zeros(inputlen)
        self.weights = np.random.uniform(weightinit[0], weightinit[1], size=(neurons, inputlen)).round(1)
        self.biases = np.random.uniform(biasinit[0], biasinit[1], neurons).round(1)
        self.output = np.zeros(neurons)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            f"Layer class initialized: {inputlen} inputs; {neurons} nodes; random weights between {weightinit}; random biases between {biasinit}")

    @log_exceptions
    def feed_input(self, input: Sequence[float] | npt.NDArray[np.float64] | Self) -> npt.NDArray[np.float64]:
        """
        Populate the layer's input vector.

        Args:
            input (Sequence[float] | npt.NDArray[np.float64] | Self): A sequence or Layer from which to extract the inputs. If a Layer is provided, the input is extracted from its output.

        Raises:
            TypeError: only sequences or Layer instances are accepted
            ValueError: the shape of the input parameter must match the shape of the Layer's input attribute

        Side Effects:
            self.input is populated with the method's return value

        Returns:
            a numpy array with the layer's inputs
        """
        if isinstance(input, Sequence) or isinstance(input, np.ndarray):
            input_vector = np.array(input)
        elif isinstance(input, Layer):
            input_vector = input.output
        else:
            raise TypeError(f"Wrong input type: {type(input)}")
        if len(input_vector) != len(self.input):
            raise ValueError(f"Bad input shape. Expected {len(self.input)}, got {len(input_vector)}")
        self.input = input_vector
        self.logger.debug(f"New layer inputs: {input_vector}")
        return self.input

    @log_exceptions
    def calculate_output(self, activation_func: Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None) -> npt.NDArray[np.float64]:
        """
        Calculate the layer's output - the sum of the products of inputs and weights + bias, optionally wrapped in an activation function.

        Args:
            activation_func: If provided, the result is passed to it before the method returns. Defaults to None.

        Side Effects:
            self.output is populated with the calculation result

        Returns:
            the calculation result
        """
        result = np.dot(self.weights, self.input) + self.biases
        if activation_func:
            result = activation_func(result)
        self.output = result
        self.logger.debug(f"Layer calculation result: {result}")
        return result


class NeuralNetwork:
    """
    Implementation of a neural network consisting of an input vector and an array of Layer instances.

    Args:
        inputs (int): the number of inputs this network takes
    """

    def __init__(self, inputs: int) -> None:
        self.input = np.zeros(inputs)
        self.layers: list[Layer] = []
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"NeuralNetwork class initialized: {inputs} inputs")

    @log_exceptions
    def feed_input(self, input: Sequence[float] | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Populate the network's input vector.

        Args:
            input (Sequence[float] | npt.NDArray[np.float64]): The sequence to be added to the input vector

        Raises:
            TypeError: only sequences are accepted
            ValueError: the shape of the input parameter must match the shape of the network's input attribute

        Side Effects:
            self.input is populated with the method's return value

        Returns:
            a numpy array with the layer's inputs
        """
        if not isinstance(input, Sequence) and not isinstance(input, np.ndarray):
            raise TypeError(f"Expected array-like input, but got {type(input)}")
        if len(input) != len(self.input):
            raise ValueError(f"Bad input shape. Expected {len(self.input)}, got {len(input)}")
        input_vector = np.array(input)
        self.logger.debug(f"New network inputs: {input_vector}")
        self.input = input_vector
        return input_vector

    @log_exceptions
    def add_layer(self, nodes: int, weightinit: tuple[float, float] = (-1.0, 1.0), biasinit: tuple[float, float] = (-1.0, 1.0)) -> Layer:
        """
        Adds a new layer to the network, setting its number of inputs to the number of nodes/outputs of the previous layer.

        Args:
            nodes (int): the number of nodes of the new layer
            weightinit (tuple[float, float], optional): The range within which to initialize the layer's weights. Defaults to (-1.0, 1.0).
            biasinit (tuple[float, float], optional): The range within which to initialize the layer's biases. Defaults to (-1.0, 1.0)

        Side Effects:
            Appends the new Layer instance to self.layers

        Returns:
            the new Layer instance
        """
        if not self.layers:
            inputs = len(self.input)
        else:
            inputs = len(self.layers[-1].output)
        new_layer = Layer(inputs, nodes, weightinit, biasinit)
        self.layers.append(new_layer)
        self.logger.debug(f"Added new layer to NeuralNetwork. Current number of layers: {len(self.layers)}")
        return new_layer

    @log_exceptions
    def forward_pass(self, activation_func: Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None) -> npt.NDArray[np.float64]:
        """
        Chain the calculation of each Layer, feeding one Layer's output into the input of the next one, optionally wrapping them into an activation function.

        Args:
            activation_func: If provided, the result is passed to it before feeding it into the next Layer's input. Defaults to None.

        Returns:
            the output of the final Layer
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                inputs = self.input
            else:
                inputs = self.layers[i-1].output
            layer.feed_input(inputs)
            layer.calculate_output(activation_func)
        output = self.layers[-1].output
        self.logger.debug(f"Forward pass completed. Output: {output}")
        return output

import numpy as np
import numpy.typing as npt
from typing import Callable, Self, Optional, Any
import logging
from utils.logger import log_exceptions


class Layer:
    """
    Implementation of an individual layer of a neural network

    Initialization args:
        inputlen (int): the number of inputs the layer takes
        neurons (int): the number of nodes of the layer
        batchsize (int): the size of the input batch
        weightinit (tuple[float, float], optional): The range within which to initialize the layer's weights. Defaults to (-1.0, 1.0).
        biasinit (tuple[float, float], optional): The range within which to initialize the layer's biases. Defaults to (-1.0, 1.0).
    """

    def __init__(self, inputlen: int, neurons: int, batchsize: int, weightinit: tuple[float, float] = (-1.0, 1.0), biasinit: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.input = np.zeros((batchsize, inputlen))
        self.weights = np.random.uniform(weightinit[0], weightinit[1], size=(neurons, inputlen)).round(1)
        self.biases = np.random.uniform(biasinit[0], biasinit[1], neurons).round(1)
        self.output = np.zeros((batchsize, neurons))
        self.logger = logging.getLogger('neuralnet.'+__name__)
        self.logger.debug(
            f"Layer class initialized: {inputlen} inputs; {neurons} nodes; random weights between {weightinit}; random biases between {biasinit}")

    @log_exceptions
    def feed_input(self, input: npt.NDArray[np.float64] | Self) -> npt.NDArray[np.float64]:
        """
        Populate the layer's input matrix.

        Args:
            input: A 2d array or Layer from which to extract the inputs. If a Layer is provided, the input is extracted from its output.

        Raises:
            TypeError: only sequences or Layer instances are accepted
            ValueError: the shape of the input parameter must match the shape of the Layer's input attribute

        Side Effects:
            self.input is populated with the method's return value

        Returns:
            a numpy array with the layer's inputs
        """
        if isinstance(input, np.ndarray):
            input_matrix = input
        elif isinstance(input, Layer):
            input_matrix = input.output
        else:
            raise TypeError(f"Wrong input type: {type(input)}")
        if input_matrix.shape != self.input.shape:
            raise ValueError(f"Bad input shape. Expected {self.input.shape}, got {input_matrix.shape}")
        self.input = input_matrix
        self.logger.debug(f"New layer inputs:\n{input_matrix}")
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
        result = np.dot(self.input, self.weights.T) + self.biases
        if activation_func:
            result = activation_func(result)
        self.output = result
        self.logger.debug(f"Layer calculation result:\n{result}")
        return result


class NeuralNetwork:
    """
    Implementation of a neural network consisting of an input matrix and an array of Layer instances.

    Args:
        inputs (int): the number of inputs this network takes
        batchsize (int): the size of the input batch
    """

    def __init__(self, inputs: int, batchsize: int) -> None:
        self.input = np.zeros((batchsize, inputs))
        self.layers: list[Layer] = []
        self.logger = logging.getLogger('neuralnet.'+__name__)
        self.logger.debug(f"NeuralNetwork class initialized: {inputs} inputs; {batchsize} batchsize.")

    @log_exceptions
    def feed_input(self, input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Populate the network's input matrix.

        Args:
            input: The 2d array to be added to the input matrix

        Raises:
            TypeError: only sequences are accepted
            ValueError: the shape of the input parameter must match the shape of the network's input attribute

        Side Effects:
            self.input is populated with the method's return value

        Returns:
            a numpy 2d array with the layer's inputs
        """
        if not isinstance(input, np.ndarray):
            raise TypeError(f"Expected numpy array, but got {type(input)}")
        if input.shape != self.input.shape:
            raise ValueError(f"Bad input shape. Expected {self.input.shape}, got {input.shape}")
        input_matrix = np.array(input)
        self.logger.debug(f"New network inputs:\n{input_matrix}")
        self.input = input_matrix
        return input_matrix

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
            inputs = self.input.shape[1]
        else:
            inputs = self.layers[-1].output.shape[1]
        new_layer = Layer(inputs, nodes, self.input.shape[0], weightinit, biasinit)
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
        self.logger.debug(f"Forward pass completed. Output:\n{output}")
        return output

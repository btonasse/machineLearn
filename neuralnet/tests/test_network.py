import unittest
import numpy as np
from network import NeuralNetwork, Layer
from network import activation
import logging

logger = logging.getLogger()
logger.addHandler(logging.NullHandler(logging.DEBUG))


class LayerTest(unittest.TestCase):
    """
    Test Layer class initialization and API
    """

    def setUp(self) -> None:
        np.random.seed(0)
        self.layer = Layer(3, 5, 1)
        self.inputs = np.array([[1.0, 2.0, 3.0]])
        self.fixed_weights = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.fixed_biases = np.array([1, 2, 1, 1, 1])

    def test_layer_initialization(self):
        """Random initialization of weights and biases"""
        expected_weights = [
            [0.1,  0.4,  0.2],
            [0.1, -0.2,  0.3],
            [-0.1,  0.8,  0.9],
            [-0.2,  0.6,  0.1],
            [0.1,  0.9, -0.9]]

        expected_biases = [-0.8, -1.0,  0.7,  0.6,  0.7]

        self.assertTrue(np.allclose(self.layer.weights, expected_weights))
        self.assertTrue(np.allclose(self.layer.biases, expected_biases))

    def test_feed_input(self):
        """Feeding different types of input to a layer"""
        self.layer.feed_input(self.inputs)
        self.assertTrue(np.array_equal(self.inputs, self.layer.input))

        input_array_1d = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.layer.feed_input(input_array_1d)

        input_layer = Layer(2, 3, 1)
        input_layer.output = self.inputs
        self.layer.feed_input(input_layer)
        self.assertTrue(np.array_equal(input_layer.output, self.layer.input))

        input_layer_bad_shape = Layer(2, 4, 1)
        input_layer_bad_shape.output = np.array([[1.0, 2.0, 3.0, 4.0]])
        with self.assertRaises(ValueError):
            self.layer.feed_input(input_layer_bad_shape)

    def test_calculate_output(self):
        """Layer output calculation"""
        self.layer.feed_input(self.inputs)
        self.layer.weights = self.fixed_weights
        self.layer.biases = self.fixed_biases
        expected_output = np.array([[15, 16, 15, 15, 15]])
        output = self.layer.calculate_output()
        self.assertTrue(np.array_equal(expected_output, output), f"Output:\n{output}")

    def test_relu(self):
        """Layer output calculation with relu"""
        self.layer.feed_input(self.inputs)
        self.layer.weights = np.array([[1, 2, -3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.layer.biases = self.fixed_biases
        self.layer.activation_func = activation.relu
        expected_output = np.array([[0, 16, 15, 15, 15]])
        output = self.layer.calculate_output()
        self.assertTrue(np.array_equal(expected_output, output), f"Output:\n{output}")


class NeuralNetworkTest(unittest.TestCase):
    """
    Test building a NeuralNetwork instance and its APIs
    """

    def setUp(self) -> None:
        self.network = NeuralNetwork(3, 1)

    def test_add_layer(self):
        """Adding a new layer to Network"""
        self.network.add_layer(5)
        self.network.add_layer(5)
        self.network.add_layer(2)
        self.assertTrue(self.network.layers[0].input.shape == (1, 3))
        self.assertTrue(self.network.layers[0].output.shape == (1, 5))
        self.assertTrue(self.network.layers[1].input.shape == (1, 5))
        self.assertTrue(self.network.layers[1].output.shape == (1, 5))
        self.assertTrue(self.network.layers[2].input.shape == (1, 5))
        self.assertTrue(self.network.layers[2].output.shape == (1, 2))

    def test_forward_pass(self):
        """Forward pass"""
        self.network.feed_input(np.array([[1.0, -2.0, 0.2]]))
        self.network.add_layer(5)
        self.network.add_layer(5)
        self.network.add_layer(2)
        expected = np.array([[0.1856, 0.7844]])
        actual = self.network.forward_pass()
        self.assertTrue(np.allclose(expected, actual))

        expected_relu = np.array([0, 1.986])
        for layer in self.network.layers:
            layer.activation_func = activation.relu
        actual_relu = self.network.forward_pass()
        self.assertTrue(np.allclose(expected_relu, actual_relu))

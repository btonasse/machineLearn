import unittest
import numpy as np
from network import NeuralNetwork, Layer
from network import activation


class LayerTest(unittest.TestCase):
    """
    Test Layer class initialization and API
    """

    def setUp(self) -> None:
        np.random.seed(0)
        self.layer = Layer(3, 5)

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
        input_list = [1.0, 2.2, -0.2]
        self.layer.feed_input(input_list)
        self.assertTrue(np.array_equal(input_list, self.layer.input))

        input_numpy = np.array(input_list)
        self.layer.feed_input(input_numpy)
        self.assertTrue(np.array_equal(input_numpy, self.layer.input))

        input_layer = Layer(2, 3)
        input_layer.output = input_numpy
        self.layer.feed_input(input_layer)
        self.assertTrue(np.array_equal(input_layer.output, self.layer.input))

        input_layer_bad_shape = Layer(2, 4)
        input_layer_bad_shape.output = np.array([1.0, 2.2, -0.2, 3.0])
        with self.assertRaises(ValueError):
            self.layer.feed_input(input_layer_bad_shape)

    def test_calculate_output(self):
        """Layer output calculation"""
        self.layer.feed_input([1, 2, 3])
        self.layer.weights = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.layer.biases = np.array([1, 2, 1, 1, 1])
        expected_output = np.array([15, 16, 15, 15, 15])
        output = self.layer.calculate_output()
        self.assertTrue(np.array_equal(expected_output, output))

    def test_relu(self):
        """Layer output calculation with relu"""
        self.layer.feed_input([1, 2, 3])
        self.layer.weights = np.array([[1, 2, -3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.layer.biases = np.array([1, 2, 1, 1, 1])
        expected_output = np.array([0, 16, 15, 15, 15])
        output = self.layer.calculate_output(activation.relu)
        self.assertTrue(np.array_equal(expected_output, output))


class NeuralNetworkTest(unittest.TestCase):
    """
    Test building a NeuralNetwork instance and its APIs
    """

    def setUp(self) -> None:
        self.network = NeuralNetwork(3)

    def test_add_layer(self):
        """Adding a new layer to Network"""
        self.network.add_layer(5)
        self.network.add_layer(5)
        self.network.add_layer(2)
        self.assertTrue(len(self.network.layers[0].input) == 3)
        self.assertTrue(len(self.network.layers[0].output) == 5)
        self.assertTrue(len(self.network.layers[1].input) == 5)
        self.assertTrue(len(self.network.layers[1].output) == 5)
        self.assertTrue(len(self.network.layers[2].input) == 5)
        self.assertTrue(len(self.network.layers[2].output) == 2)

    def test_forward_pass(self):
        """Forward pass"""
        self.network.feed_input([1, -2, 0.2])
        self.network.add_layer(5)
        self.network.add_layer(5)
        self.network.add_layer(2)
        expected = np.array([0.1856, 0.7844])
        actual = self.network.forward_pass()
        self.assertTrue(np.allclose(expected, actual))

        expected_relu = np.array([0, 1.986])
        actual_relu = self.network.forward_pass(activation.relu)
        self.assertTrue(np.allclose(expected_relu, actual_relu))

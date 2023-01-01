import unittest
import numpy as np
from network import NeuralNetwork, Layer


class NeuralNetworkTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.layer = Layer(3, 5)

    def test_layer_initialization(self):
        expected_weights = [
            [0.09762701, 0.43037873, 0.20552675],
            [0.08976637, -0.1526904, 0.29178823],
            [-0.12482558, 0.783546, 0.92732552],
            [-0.23311696, 0.58345008, 0.05778984],
            [0.13608912, 0.85119328, -0.85792788]
        ]
        expected_biases = [-0.8257414, -0.95956321, 0.66523969, 0.5563135, 0.7400243]

        self.assertTrue(np.allclose(self.layer.weights, expected_weights))
        self.assertTrue(np.allclose(self.layer.biases, expected_biases))

    def test_feed_input(self):

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
        pass

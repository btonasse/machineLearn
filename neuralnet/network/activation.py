"""
Define activation functions to be used in the NeuralNetwork calculations
"""

import numpy as np
import numpy.typing as npt


def relu(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Takes a 1D vector and applies the Rectified Linear Unit function (return 0 if element is less than 0) to each element.
    """
    return np.maximum(0, vector)


def sigmoid(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    raise NotImplementedError

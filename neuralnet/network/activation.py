"""
Define activation functions to be used in the NeuralNetwork calculations
"""

import numpy as np
import numpy.typing as npt


def relu(batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Takes a 2D matrix and applies the Rectified Linear Unit function (return 0 if element is less than 0) to each element.
    """
    return np.maximum(0, batch)


def sigmoid(batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    raise NotImplementedError

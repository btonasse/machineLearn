import numpy as np
import numpy.typing as npt


def relu(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(0, vector)


def sigmoid(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    raise NotImplementedError

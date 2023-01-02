"""
Main entry point of the program.

Sets up the root logger, logging DEBUG and above to a file
"""

import logging
from network import NeuralNetwork
import numpy as np

logger = logging.getLogger('neuralnet')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs.log', mode='w')
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.debug("Logger initialized.")


def main():
    print('Initializing neuralnet...')
    # Quick test
    net = NeuralNetwork(3, 2)
    net.add_layer(4)
    net.add_layer(4)
    net.add_layer(1)
    net.feed_input(np.array([[23.0, -2.0, 0.45], [25.0, 2.0, 12.3]]))
    result = net.forward_pass()
    print(result)


if __name__ == '__main__':
    main()

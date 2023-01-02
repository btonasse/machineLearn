"""
Main entry point of the program.

Sets up the root logger, logging DEBUG and above to a file
"""

import logging
from network import NeuralNetwork

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs.log', mode='w')
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

"""console_handler = logging.StreamHandler()
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.WARN)
logger.addHandler(console_handler)"""


def main():
    print('Initializing neuralnet...')
    # Quick test
    net = NeuralNetwork(3)
    net.add_layer(4)
    net.add_layer(4)
    net.add_layer(1)
    net.feed_input([23, -2, 0.45])
    result = net.forward_pass()
    print(result)


if __name__ == '__main__':
    main()

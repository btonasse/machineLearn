"""
Provides utility functions related to logging
"""

import functools
from typing import Callable


def log_exceptions(func: Callable):
    """
    Decorator that wraps a function in a try-except block and logs any exceptions before reraising them.

    Args:
        func (callable): the function to decorate
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            args[0].logger.exception(e)
            raise
    return wrapper

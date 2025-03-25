
import time
import logging

code_Green = '\033[92m'
code_blue = '\033[94m'
code_end = '\033[0m'


def measure(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{code_blue}Execution time of {
                     func.__name__}: {end - start}{code_end}")
        return result, end - start
    return wrapper

from time import time


def timing(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print("Time : {:.10f} s".format(end - start))
        return result

    return wrapper

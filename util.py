from time import time


def timing(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print("Time : {:.10f} s".format(end - start))
        return result

    return wrapper


def avg_timing(times, func, *args):
    start = time()
    for i in range(times):
        func(*args)
    end = time()
    return (end - start) / times

# 是否启用计时装饰器
count = True

if not count:
    timing = lambda x: x
from time import time

# 是否启用计时装饰器
count = True


def timing(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        if count:
            print("Time : {:.10f} s".format(end - start))
        return result

    return wrapper


def avg_timing(times, func, *args):
    start = time()
    for _ in range(times):
        func(*args)
    end = time()
    return (end - start) / times


def sum_timing(times, func, *args):
    start = time()
    for _ in range(times):
        func(*args)
    end = time()
    return end - start
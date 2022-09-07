import time


def timer(description: str):
    def kernel(func):
        def wrapper(*args, **kw):
            tick = time.time()
            res = func(*args, **kw)
            print(f"{description} runs for {time.time() - tick:.3f}s")
            return res
        return wrapper
    return kernel

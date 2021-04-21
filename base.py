import os
import pickle
import threading
from functools import wraps

CACHE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
CACHE = threading.local()


def cache(cache_name):
    def caching_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            cache_file = os.path.join(CACHE_ROOT, cache_name + ".pickle")

            if hasattr(CACHE, cache_name):
                return getattr(CACHE, cache_name)
            elif os.path.exists(cache_file):
                with open(cache_file, "rb") as fp:
                    data = pickle.load(fp)

                setattr(CACHE, cache_name, data)
                return data

            data = func(*args, **kwargs)

            with open(cache_file, "wb") as fp:
                pickle.dump(data, fp)

            setattr(CACHE, cache_name, data)
            return data

        return wrapped_function

    return caching_decorator

from functools import wraps

def default_args(defaults):
    """ return a decorator that takes a function as input """
    def wrapper(func):
        @wraps(func)  # just to show docstring of original function
        def new_func(*args, **kwargs):
            kwargs = defaults | kwargs
            return func(*args, **kwargs)
        return new_func
    return wrapper

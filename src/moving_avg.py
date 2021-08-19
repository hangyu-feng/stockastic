import numpy as np

def sma(values):
    """ simple moving average """
    return np.mean(values)

def ema(values, k=None):
    """ expoential moving average:
    k = 2 / (len(values) + 1), can actually set to other numbers
    ema[i] = closing_price[i] * k + ema[i-1] * (1-k)
    can be calculated as weighted sum:
        sum((1-k)**i * v[i]) / sum((1-k)**i), where i = 0...(N-1), v[0] is today's value

    https://en.wikipedia.org/wiki/Moving_average
    """
    N = len(values)
    if k is None: k = 2 / (N+1)
    weights = np.power(1-k, range(N))
    S = (1 - (1-k)**N) / k  # sum of weights
    return weights.dot(values) / S

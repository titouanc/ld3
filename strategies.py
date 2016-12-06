import numpy as np


def random(t, q):
    """Random strategy"""
    return np.random.randint(len(q))


def fixed_greedy(epsilon, t, q):
    if t == 0 or np.random.rand() < epsilon:
        return random(t, q)
    else:
        return np.argmax(q)


def fixed_softmax(tau, t, q):
    assert tau != 0
    e = np.exp(q/tau)
    p = e/sum(e)
    return (np.random.rand() < p.cumsum()).argmax()


def dynamic_greedy(t, q):
    return fixed_greedy(1 / np.sqrt(t), t, q)


def dynamic_softmax(k, tmax, t, q):
    return fixed_softmax(k * float(max(tmax - t, 1)) / tmax, t, q)

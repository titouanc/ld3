import numpy as np
import functools


def partial(func, *args, **kwargs):
    """
    Like functools.partials but adds a pretty name
    """
    res = functools.partial(func, *args, **kwargs)
    res.__name__ = func.__name__
    if args:
        res.__name__ += repr(args)
    if kwargs:
        res.__name__ += repr(kwargs)
    return res


def random(t, q, counts=None, max_q=None):
    """Random strategy"""
    return np.random.randint(len(q))


def fixed_greedy(epsilon, t, q, *args, **kwargs):
    """Greedy with fixed random epxloration"""
    if t == 0 or np.random.rand() < epsilon:
        return random(t, q)
    else:
        return np.argmax(q)


def fixed_softmax(tau, t, q, *args, **kwargs):
    """Softmax with constant temperature"""
    assert tau != 0
    e = np.exp(q/tau)
    p = e/sum(e)
    return (np.random.rand() < p.cumsum()).argmax()


def dynamic_greedy(t, q, *args, **kwargs):
    """Greedy with inverse square root decreasing random exploration"""
    return fixed_greedy(1 / np.sqrt(t), t, q)


def dynamic_softmax(k, tmax, t, q, *args, **kwargs):
    """Softmax with linearly decreasing temperature tau = k*(tmax-t)"""
    return fixed_softmax(k * float(max(tmax - t, 1)) / tmax, t, q)


def fmq(t, q, counts, max_q, c=1, s=1, tau0=4, tauN=1):
    """Frequency maximum Q heuristic"""
    # The game is stochastic, therefore the freq of the best reward is always
    # 1/(times played)
    ev = q + c * max_q / counts
    ev[counts == 0] = 0  # 0-div
    tau = np.exp(-s*t) * tau0 + tauN
    return fixed_softmax(tau, t, ev)


class Strat2D:
    """Apply a strategy on a 2-player JAL Q matrix"""

    def __init__(self, strategy):
        self.strat = strategy

    def __call__(self, t, q, counts, max_q):
        s = counts.sum()
        if s == 0:
            ca, cb = np.zeros(len(q)), np.zeros(len(q))
        else:
            ca = counts.sum(axis=0) / s  # Probability for A actions
            cb = counts.sum(axis=1) / s  # Probability for B actions

        # Average reward, given opponent actions distribution
        qa = (q*cb.reshape(len(cb), 1)).mean(axis=0)  # Vertical mul
        qb = (q*ca).mean(axis=1)  # Horizontal mul
        sa = self.strat(t, qa, counts.sum(axis=0), max_q.max(axis=0))
        sb = self.strat(t, qb, counts.sum(axis=1), max_q.max(axis=1))
        return sb, sa

    @property
    def __name__(self):
        return "2D " + self.strat.__name__

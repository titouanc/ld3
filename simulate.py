import functools
from multiprocessing import Pool, cpu_count, current_process
from time import time
import numpy as np


# Divisions by 0 are handled; discard warnings
np.seterr('ignore')


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


def qlearn(t, q, rewards, strategy):
    """
    Perform a single action, at time `t` and with Q-value `q`. Each action has
    a reward, and `strategy` is a function Time -> Qvalues -> ActionNo.
    Return the chosen action and the new Q-values vector
    """
    assert q.shape[1:] == rewards.shape

    # Estimate the average reward with arithmetic mean
    avg_reward = q[1] / q[0]
    avg_reward[q[0] == 0] = 0  # Overwrite div-by-0 errors
    i = strategy(t, avg_reward)
    ii = (i,) if '__iter__' not in dir(i) else i
    res = q.copy()
    res[(0,) + ii] += 1
    res[(1,) + ii] += rewards[i]
    return i, res


def simulate(mu, sigma, strategy, epochs):
    """
    Perform a whole simulation for a single agent for a determined amount of
    `epochs`, and actions rewards of mean `mu` and stddev `sigma`. The number
    of actions is the length of mu and sigma.
    `strategy` is a function [Time -> Q-values -> Action-Number].
    """
    assert mu.shape == sigma.shape

    q = np.zeros((epochs, 2) + mu.shape)
    r = np.zeros(epochs)
    a = np.zeros(epochs)
    rewards = np.random.normal(loc=mu, scale=sigma, size=(epochs,)+mu.shape)

    for t in range(epochs):
        a[t], q[t] = qlearn(t, q[max(0, t-1)], rewards[t], strategy)
        r[t] = rewards[t, a[t]]

    return r, a


def _simulate(_, *args, **kwargs):
    """
    A wrapper around simulate that simply uncurries its first argument;
    used for multiprocessing.Pool.map. Also avoid getting the random state
    inherited from parent process
    """
    np.random.seed(int(100000 * time() * current_process().pid) & 0xffffffff)
    return simulate(*args, **kwargs)


def multisim(mu, sigma, strategy, epochs, runs):
    """
    Run simulate in parallel for `runs` different agents. Same arguments as
    simulate.
    """
    sim = partial(_simulate, mu=mu, sigma=sigma,
                  strategy=strategy, epochs=epochs)
    workers = Pool(cpu_count())
    return np.array(zip(*workers.map(sim, range(runs))))
    # return np.array(zip(*map(sim, range(runs))))

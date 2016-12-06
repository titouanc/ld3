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
    assert len(q) == len(rewards)
    # Estimate the average reward with arithmetic mean
    avg_reward = (q[:, 1] / q[:, 0])
    avg_reward[(q[:, 0] == 0)] = 0  # Overwrite div-by-0 errors
    i = strategy(t, avg_reward)
    res = 1 * q
    res[i] += [1, rewards[i]]
    return i, res


def simulate(mu, sigma, strategy, epochs=1000):
    """
    Perform a whole simulation for a single agent for a determined amount of
    `epochs`, and actions rewards of mean `mu` and stddev `sigma`. The number
    of actions is the length of mu and sigma.
    `strategy` is a function [Time -> Q-values -> Action-Number].
    """
    assert len(mu) == len(sigma)

    q = np.zeros((epochs, len(mu), 2))
    r = np.zeros(epochs)
    a = np.zeros(epochs)

    for t in range(epochs):
        rewards = np.random.normal(loc=mu, scale=sigma)
        a[t], q[t] = qlearn(t, q[max(0, t-1)], rewards, strategy)
        r[t] = rewards[a[t]]

    return r, a


def _simulate(_, *args, **kwargs):
    """
    A wrapper around simulate that simply uncurries its first argument;
    used for multiprocessing.Pool.map. Also avoid getting the random state
    inherited from parent process
    """
    np.random.seed(int(100000 * time() * current_process().pid))
    return simulate(*args, **kwargs)


def multisim(mu, sigma, strategy, epochs=1000, runs=1000):
    """
    Run simulate in parallel for `runs` different agents. Same arguments as
    simulate.
    """
    workers = Pool(cpu_count())
    sim = partial(_simulate, mu=mu, sigma=sigma,
                  strategy=strategy, epochs=epochs)
    return np.array(zip(*workers.map(sim, range(runs))))

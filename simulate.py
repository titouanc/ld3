from functools import partial
from multiprocessing import Pool, cpu_count, current_process
from time import time
import numpy as np
from local_config import PARALLEL, RUNS


# Divisions by 0 are handled; discard warnings
np.seterr('ignore')


def qlearn(t, q, max_q, rewards, strategy):
    """
    Perform a single action, at time `t` and with Q-value `q`. Each action has
    a reward, and `strategy` is a function Time -> Qvalues -> ActionNo.
    Return the chosen action and the new Q-values vector
    """
    assert q.shape[1:] == rewards.shape

    # Estimate the average reward with arithmetic mean
    avg_reward = q[1] / q[0]
    avg_reward[q[0] == 0] = 0  # Overwrite div-by-0 errors
    i = strategy(t, avg_reward, q[0], max_q)
    res = q.copy()
    res[0][i] += 1
    res[1][i] += rewards[i]
    if rewards[i] > max_q[i]:
        max_q[i] = rewards[i]
    return i, res, max_q


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
    a = np.zeros((epochs, len(mu.shape)))
    rewards = np.random.normal(loc=mu, scale=sigma, size=(epochs,)+mu.shape)
    max_q = np.zeros(mu.shape)

    for t in range(epochs):
        a[t], q[t], max_q = qlearn(t, q[max(0, t-1)], max_q,
                                   rewards[t], strategy)
        r[t] = rewards[t][tuple(a[t])]

    return q, r, a


def _simulate(_, *args, **kwargs):
    """
    A wrapper around simulate that simply uncurries its first argument;
    used for multiprocessing.Pool.map. Also avoid getting the random state
    inherited from parent process
    """
    if PARALLEL:
        s = int(100000 * time() * current_process().pid)
        np.random.seed(s & 0xffffffff)
    return simulate(*args, **kwargs)


def multisim(mu, sigma, strategy, epochs, runs=RUNS):
    """
    Run simulate in parallel for `runs` different agents. Same arguments as
    simulate.
    """
    sim = partial(_simulate, mu=mu, sigma=sigma,
                  strategy=strategy, epochs=epochs)
    if PARALLEL:
        workers = Pool(cpu_count())
        res = workers.map(sim, range(runs))
    else:
        res = map(sim, range(runs))
    Q, R, A = zip(*res)
    return tuple(np.array(x) for x in (Q, R, A))


def multistrat(mu, sigma, strategies, epochs, runs=RUNS):
    """
    Like multisim, but for multiple strategies
    """
    sim = partial(multisim, mu=mu, sigma=sigma, epochs=epochs, runs=runs)
    Q, R, A = zip(*[sim(strategy=s) for s in strategies])
    return tuple(np.array(x) for x in (Q, R, A))

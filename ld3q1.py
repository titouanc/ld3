import numpy as np
from matplotlib.gridspec import GridSpec
from time import time
from plot import plt, show

from simulate import multistrat, multisim, simulate
from strategies import (
    partial, random,
    fixed_greedy, dynamic_greedy,
    fixed_softmax, dynamic_softmax)


# Hi-lvl helpers
def multiplot(A, R, strats):
    colors = 'kgrcmyb'
    nA = int(A.max() + 1)
    spec = GridSpec(nA+1, 2, width_ratios=[4, 1], height_ratios=[3]+[1]*nA)
    fig = plt.figure(figsize=(12, 12))

    ax = fig.add_subplot(spec[0])
    for s, strat in enumerate(strats):
        plt.plot(R[s].mean(axis=0), label=strat.__name__, alpha=.4, c=colors[s])
    ax.set_xlabel("Epoch")
    plt.ylabel("Average reward")
    plt.title("Average reward over {} runs".format(len(A[0])))
    plt.legend(loc="lower right", fontsize=(10))

    axx = ax
    for i in range(nA):
        axx = fig.add_subplot(spec[2+2*i], sharex=axx)
        for s, strat in enumerate(strats):
            plt.plot(100*(A[s, :, :] == i).mean(axis=0), c=colors[s], alpha=.4)
        plt.ylabel('% Act. {}'.format(i))

    fig.add_subplot(spec[1], sharey=ax)
    bp = plt.boxplot(R.mean(axis=2).T)
    for box, color in zip(bp['boxes'], colors):
        box.set_color(color)
    plt.xticks([])
    plt.title("Average\nreward")

    fig.add_subplot(spec[1:-1, -1], sharey=ax)
    bp = plt.boxplot(R[:,:,-100:].mean(axis=2).T, labels=[x.__name__ for x in strats])
    for box, color in zip(bp['boxes'], colors):
        box.set_color(color)
    plt.xticks(rotation='vertical', fontsize=(10))
    plt.title("Last 100 epochs")


def demo(name, mu, sigma, strategies):
    t0 = time()
    R, A = multistrat(mu=mu, sigma=sigma,
                      strategies=strategies, epochs=1100)
    multiplot(A, R, strategies)
    show(name.replace(' ', '_'))
    print name, "ran in", time()-t0, "s"


if __name__ == "__main__":
    mu, sigma = np.array([[2.3, 2.1, 1.5, 1.3], [0.9, 0.6, 0.4, 2]])

    static_strategies = [
        random,
        partial(fixed_greedy, 0),
        partial(fixed_greedy, 0.1),
        partial(fixed_greedy, 0.2),
        partial(fixed_softmax, 1),
        partial(fixed_softmax, 0.1),
    ]
    demo("Ex 1_1", mu, sigma, static_strategies)
    demo("Ex 1_2", mu, 2*sigma, static_strategies)

    dynamic_strategies = [
        random,
        dynamic_greedy,
        partial(dynamic_softmax, 4, 1000)
    ]
    demo("Ex 1_3_1", mu, sigma, dynamic_strategies)
    demo("Ex 1_3_2", mu, 2*sigma, dynamic_strategies)

import numpy as np
from simulate import multistrat
from strategies import partial, random, fixed_softmax, Strat2D
from plot import plt, show
from time import time

MU = np.array([[ 11,-30, 0],
               [-30,  7, 6],
               [  0,  0, 5]])
SIGMA1 = .2 * np.ones((3, 3))
SIGMA2 = .1 * np.ones((3, 3))
SIGMA2[0, 0] = 4
SIGMA3 = .1 * np.ones((3, 3))
SIGMA3[1, 1] = 4

if __name__ == "__main__":
    strats = [Strat2D(random)] + [
        Strat2D(partial(fixed_softmax, i)) for i in [1, 0.1, 0.01]
    ]

    colors = "krgb"

    for si, sigma in enumerate([SIGMA1, SIGMA2, SIGMA3]):
        t0 = time()
        R, A = multistrat(mu=MU, sigma=sigma, strategies=strats, epochs=5000)

        plt.figure(figsize=(7, 7))
        for i, s in enumerate(strats):
            plt.plot(R[i].mean(axis=0), label=s.__name__, alpha=0.5, c=colors[i])
        plt.legend(fontsize=10, loc='center right')
        plt.title("Average reward over {} runs".format(len(R[0])))
        name = "Ex_2_sigma{}".format(si+1)
        print name, time()-t0, "s"
        show(name)

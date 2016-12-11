import numpy as np
from simulate import multistrat
from strategies import partial, fixed_softmax, fmq, Strat2D
from plot import plt, show
from matplotlib.gridspec import GridSpec
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
    strats = [
        Strat2D(partial(fmq, c=1, s=1, tau0=10, tauN=.001)),
        Strat2D(partial(fmq, c=100, s=0.1, tau0=10, tauN=.001))
    ] + [
        Strat2D(partial(fixed_softmax, i)) for i in [1, 0.1, 0.01]
    ]

    spec = GridSpec(1, 2, width_ratios=[4, 1])

    colors = "rgbky"

    for si, sigma in enumerate([SIGMA1, SIGMA2, SIGMA3]):
        t0 = time()
        Q, R, A = multistrat(mu=MU, sigma=sigma,
                             strategies=strats, epochs=5000)

        fig = plt.figure(figsize=(12, 5))

        ax = fig.add_subplot(spec[0])
        for i, s in enumerate(strats):
            plt.plot(R[i].mean(axis=0), label=s.__name__, alpha=0.5, c=colors[i])
        plt.legend(fontsize=10, loc="lower right")
        plt.title("Average reward over {} runs".format(len(R[0])))

        fig.add_subplot(spec[1], sharey=ax)
        bp = plt.boxplot(R.mean(axis=2).T, labels=[s.__name__ for s in strats])
        for box, color in zip(bp['boxes'], colors):
            box.set_color(color)
        plt.xticks([])
        plt.ylim(3)

        name = "Ex_2_sigma{}".format(si+1)
        print name, time()-t0, "s"
        show(name)

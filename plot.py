from local_config import INTERACTIVE


if not INTERACTIVE:
    # Backend without X server
    import matplotlib as mpl
    mpl.use('Agg')


import matplotlib.pyplot as plt


def show(figname="figure"):
    if INTERACTIVE:
        plt.show()
    else:
        plt.savefig(figname + ".pdf")

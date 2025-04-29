from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
from scipy.linalg import eigh as scipy_eigh
import numpy as np
import torch


def gen():
    iris = fetch_ucirepo(id=53)

    _, ys = np.unique(iris.data.targets, return_inverse=True)
    ys = torch.from_numpy(ys)

    xs = torch.from_numpy(iris.data.features.to_numpy())
    xs -= xs.mean(0, False)
    xs /= xs.std(0, False)

    t_cov = torch.cov(xs.T)
    _, e_vecs = scipy_eigh(t_cov.numpy())

    return torch.tensor(e_vecs), xs, ys


def plot_2d(e_vecs, xs, ys):
    ab = xs @ e_vecs
    plt.scatter(ab[:, -1], ab[:, -2], c=ys, s=100, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    e_vecs, xs, ys = gen()
    plot_2d(e_vecs, xs, ys)
    print("done.")
    
    
from pathlib import Path

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

from formulations.helpers import get_collocation_points

class ObserveDataPlotter:

    @classmethod
    def from_file(cls, mat_file: str):
        mat_file = str(mat_file)
        if not Path(mat_file).is_file():
            raise ValueError(f".mat file not exist: {mat_file}")
        return cls(loadmat(mat_file))

    def __init__(self, data: np.array):
        self.data = data

    def plot(self, out_f: str, overwrite: bool = True):
        out_f = str(out_f)
        if (not overwrite) and Path(out_f).is_file():
            raise FileExistsError(f"file {out_f} already exists")

        x_star = self.data["x"]
        u_star = self.data["u"]
        h_star = self.data["h"]

        fig = plt.figure(figsize=[11.7, 8.3]) # A4 size
        ax = plt.subplot(221)
        ax.set_title("x_star")
        ax.scatter(np.arange(x_star.size), x_star, s=0.1)

        ax = plt.subplot(222)
        ax.set_title("u_star")
        ax.scatter(np.arange(u_star.size), u_star, s=0.1)

        ax = plt.subplot(223)
        ax.scatter(np.arange(h_star.size), h_star, s=0.1)
        ax.set_title("h_star")
        plt.savefig(out_f)

        ax = plt.subplot(224)
        collo_pts = get_collocation_points(x_train=np.transpose(x_star), xmin=x_star.min(), xmax=x_star.max(), N_t=201)
        collo_pts = collo_pts.numpy()
        ax.scatter(np.arange(collo_pts.size), collo_pts, s=0.1)
        ax.set_title("collocation pts")
        plt.savefig(out_f)

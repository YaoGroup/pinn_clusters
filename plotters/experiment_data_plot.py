from pathlib import Path

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

class ExperimentDataPlot:

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

        fields = [
            "x_star", "x_collocation", "u_star", "h_star", "u_star_noise", "h_star_noise",
            "u_predict", "h_predict", "b_predict",
            "loss", "loss_equation", "loss_data", "residue_momentum", "residue_mass"
        ]
        data = {}
        for field in fields:
            data[field] = self.data[field]

        fig = plt.figure(figsize=[11.7, 8.3]) # A4 size
        ax = plt.subplot(221)
        ax.set_title("velocity u")
        ax.scatter(self.data["x_star"], self.data["u_star"], s=0.1)
        ax.scatter(self.data["x_star"], self.data["u_star_noise"], s=0.1)
        ax.scatter(self.data["x_star"], self.data["u_predict"], s=0.1)

        ax = plt.subplot(222)
        ax.set_title("height h")
        ax.scatter(self.data["x_star"], self.data["h_star"], s=0.1)
        ax.scatter(self.data["x_star"], self.data["h_star_noise"], s=0.1)
        ax.scatter(self.data["x_star"], self.data["h_predict"], s=0.1)

        ax = plt.subplot(223)
        ax.scatter(self.data["x_star"], self.data["b_predict"], s=0.1)
        ax.set_title("hardness b")

        ax = plt.subplot(224)
        ax.scatter(np.arange(self.data["x_collocation"].size), self.data["x_collocation"], s=0.1)
        ax.set_title("collocation pts")
        plt.savefig(out_f)


class ExprimentDataWoNoisePlot:

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

        fields = [
            "x_star", "x_collocation", "u_star", "h_star",
            "u_predict", "h_predict", "b_predict",
            "loss", "loss_equation", "loss_data", "residue_momentum", "residue_mass"
        ]
        data = {}
        for field in fields:
            data[field] = self.data[field]

        fig = plt.figure(figsize=[11.7, 8.3]) # A4 size
        ax = plt.subplot(221)
        ax.set_title("velocity u")
        ax.scatter(self.data["x_star"], self.data["u_star"], s=0.1)
        ax.scatter(self.data["x_star"], self.data["u_predict"], s=0.1)

        ax = plt.subplot(222)
        ax.set_title("height h")
        ax.scatter(self.data["x_star"], self.data["h_star"], s=0.1)
        ax.scatter(self.data["x_star"], self.data["h_predict"], s=0.1)

        ax = plt.subplot(223)
        ax.scatter(self.data["x_star"], self.data["b_predict"], s=0.1)
        ax.set_title("hardness b")

        ax = plt.subplot(224)
        ax.scatter(np.arange(self.data["x_collocation"].size), self.data["x_collocation"], s=0.1)
        ax.set_title("collocation pts")
        plt.savefig(out_f)

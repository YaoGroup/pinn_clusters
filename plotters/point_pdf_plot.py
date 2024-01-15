from pathlib import Path

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

class PointDistributionPlotter:

    def __init__(self, data):
        self.data = data

    def from_file(self, file: str):
        raise NotImplementedError()

    def plot(self, out_f: str, overwrite: bool = True):
        out_f = str(out_f)
        if (not overwrite) and Path(out_f).is_file():
            raise FileExistsError(f"file {out_f} already exists")

        fig = plt.figure(figsize=[11.7, 8.3]) # A4 size
        ax = plt.subplot()
        ax.hist(self.data, bins=100)
        ax.set_title("collocation point distribution")
        plt.savefig(out_f)

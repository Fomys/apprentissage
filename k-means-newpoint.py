import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster, metrics
from time import time
from tqdm import tqdm

warnings.filterwarnings("ignore")

path = './clustering-benchmark/src/main/resources/datasets/artificial/'


def load_data(dataset: str) -> np.ndarray:
    databrut = arff.loadarff(open(path + dataset + ".arff", 'r'))
    return np.array([[x[0], x[1]] for x in databrut[0]])

def extract_min_max(datanp):
    x = datanp[:, 0]
    y = datanp[:, 1]
    return min(x), max(x), min(y), max(y)


def generate_plot(dataset):
    datanp = load_data(dataset)
    # Fitting the model
    model = cluster.KMeans(n_clusters=3, init="k-means++")
    model.fit(datanp)
    min_x, max_x, min_y, max_y = extract_min_max(datanp)

    X = np.linspace(min_x, max_x, 200)
    Y = np.linspace(min_y, max_y, 200)
    L = np.zeros((X.shape[0], Y.shape[0]))
    for x, x_v in tqdm(enumerate(X)):
        for y, y_v in enumerate(Y):
            L[x][y] = model.predict([(y_v, x_v), ])[0]

    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, L)
    f0 = datanp[:, 0]
    f1 = datanp[:, 1]
    plt.scatter(f0, f1, c=model.labels_, s=8, edgecolors= "black")
    plt.savefig(f"k-means/{dataset}-extra.png")
    plt.plot()


models = ["xclara", "cassini", "3-spiral"]

for dataset in models:
    generate_plot(dataset)

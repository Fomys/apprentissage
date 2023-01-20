import os
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange
from scipy.io import arff
from sklearn import cluster, metrics
from time import time

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

print()
warnings.filterwarnings("ignore")

path = './clustering-benchmark/src/main/resources/datasets/artificial/'

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.io import arff
from sklearn import cluster, metrics
from time import time

warnings.filterwarnings("ignore")

path = './clustering-benchmark/src/main/resources/datasets/artificial/'


def load_data(dataset: str) -> np.ndarray:
    databrut = arff.loadarff(open(path + dataset + ".arff", 'r'))
    return np.array([[x[0], x[1]] for x in databrut[0]])


def plot(dataset, eps, min_samples) -> (np.ndarray, cluster.KMeans):
    data = load_data(dataset)
    dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=dbscan.labels_, s=8)
    plt.title(f"{dataset}: dbscan avec eps={eps} et min_samples={min_samples}")
    plt.savefig(f"dbscan/{dataset}-eps={eps}-min_samples={min_samples}.png", dpi=100)
    plt.clf()
    return data, dbscan


def compute(dataset, eps, min_samples):
    print(f"dataset: {dataset}")
    data = load_data(dataset)
    for k in [3, 5, 10]:
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(data)
        distances, indices = neigh.kneighbors(data)
        newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0, distances.shape[0])])
        sort = np.sort(newDistances)
        plt.title(f"Distance moyenne au {k} voisins pour {dataset}")
        plt.plot(sort)
        plt.xlabel(f"Nombre de points ayant une distance aux {k} voisin inférieure")
        plt.ylabel(f"Distance aux {k} voisins")
        plt.savefig(f"dbscan/{dataset}-{k}-voisins.png", dpi=100)
        plt.clf()

    t1 = time()
    data, dbscan = plot(dataset, eps, min_samples)
    t2 = time()


models = [("xclara", 10, 50), ("cassini", 0.16, 1), ("3-spiral", 2.5, 1)]
for d in models:
    compute(*d)


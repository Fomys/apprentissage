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

def plot(dataset, k) -> (np.ndarray, cluster.KMeans):
    data = load_data(dataset)
    kmeans = cluster.KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, s=8)
    plt.title(f"{dataset}: k-means avec {k} clusters")
    plt.savefig(f"k-means/{dataset}-k={k}.png", dpi=100)
    plt.clf()
    return data, kmeans


def compute(dataset):
    davies_bouldin = []
    calinski_harabasz = []
    silhouette = []
    duration = []
    print(f"dataset: {dataset}...")
    for k in tqdm(range(2, 10)):
        t1 = time()
        data, kmeans = plot(dataset, k)
        t2 = time()
        davies_bouldin.append((k, metrics.davies_bouldin_score(data, kmeans.labels_)))
        calinski_harabasz.append((k, metrics.calinski_harabasz_score(data, kmeans.labels_)))
        silhouette.append((k, metrics.silhouette_score(data, kmeans.labels_)))
        duration.append((k, t2-t1))
    plt.plot(*zip(*davies_bouldin))
    plt.ylabel("Score de Davies-Bouldin")
    plt.xlabel("Nombre de clusters")
    plt.title("Score de Davies-Bouldin en fonction du nombre de cluster")
    plt.savefig(f"k-means/{dataset}-davies_bouldin.png", dpi=100)
    plt.clf()
    plt.plot(*zip(*calinski_harabasz))
    plt.ylabel("Score de Calinski-Harabasz")
    plt.xlabel("Nombre de clusters")
    plt.title("Score de Calinski-Harabasz en fonction du nombre de cluster")
    plt.savefig(f"k-means/{dataset}-calinski_harabasz.png", dpi=100)
    plt.clf()
    plt.plot(*zip(*silhouette))
    plt.ylabel("Score de Silhouette")
    plt.xlabel("Nombre de clusters")
    plt.title("Score de Silhouette en fonction du nombre de cluster")
    plt.savefig(f"k-means/{dataset}-silhouette.png", dpi=100)
    plt.clf()
    plt.plot(*zip(*duration))
    return duration


times = []
models =  ["xclara", "cassini", "3-spiral", ]
for d in models:
    times.append(compute(d))

for (duration, model) in zip(times, models):
    plt.plot(*zip(*duration), label=model)
plt.xlabel("Nombre de cluster")
plt.ylabel("Temps de calcul (en s)")
plt.title("Temps de calcul pour chaque modèle en fonction du nombre de cluster")
plt.legend()
plt.savefig(f"agglomerative_clustering/time.png", dpi=100)
plt.clf()
#

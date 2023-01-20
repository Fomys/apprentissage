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


def plot(dataset, k, method) -> (np.ndarray, cluster.KMeans):
    data = load_data(dataset)
    agglomerative_clustering = cluster.AgglomerativeClustering(n_clusters=k, linkage=method)
    agglomerative_clustering.fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=agglomerative_clustering.labels_, s=8)
    plt.title(f"{dataset}: agglomerative_clustering avec {k} clusters")
    plt.savefig(f"agglomerative_clustering/{dataset}-{method}-k={k}.png", dpi=100)
    plt.clf()
    return data, agglomerative_clustering


def plot2(dataset, d, method) -> (np.ndarray, cluster.KMeans):
    data = load_data(dataset)
    agglomerative_clustering = cluster.AgglomerativeClustering(n_clusters=None, linkage=method, distance_threshold=d)
    agglomerative_clustering.fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=agglomerative_clustering.labels_, s=8)
    plt.title(f"{dataset}: agglomerative_clustering avec un threshold de {d}\n{agglomerative_clustering.n_clusters_} clusters trouvés")
    plt.savefig(f"agglomerative_clustering/{dataset}-{method}-d={d}.png", dpi=100)
    plt.clf()
    return data, agglomerative_clustering


def compute(dataset):
    print(f"dataset: {dataset}")
    data = load_data(dataset)
    for method in ["ward", "complete", "average", "single"]:
        linked_mat = linkage(data, method)
        dendrogram(linked_mat, orientation='top',
                   distance_sort='descending', show_leaf_counts=False)
        plt.title(f"Dendogramme du dataset {dataset}, méthode {method}")
        plt.savefig(f"agglomerative_clustering/{dataset}-dendogram-{method}.png", dpi=100)
        plt.clf()

    davies_bouldins = []
    calinski_harabaszs = []
    silhouettes = []
    durations = []
    for method in ["ward", "complete", "average", "single"]:
        davies_bouldin = []
        calinski_harabasz = []
        silhouette = []
        duration = []
        for k in range(2, 10):
            t1 = time()
            data, agglomerative_clustering = plot(dataset, k, method)
            t2 = time()
            davies_bouldin.append((k, metrics.davies_bouldin_score(data, agglomerative_clustering.labels_)))
            calinski_harabasz.append((k, metrics.calinski_harabasz_score(data, agglomerative_clustering.labels_)))
            silhouette.append((k, metrics.silhouette_score(data, agglomerative_clustering.labels_)))
            duration.append((k, t2 - t1))
        davies_bouldins.append(davies_bouldin)
        calinski_harabaszs.append(calinski_harabasz)
        silhouettes.append(silhouette)
    for (s, m) in zip(davies_bouldins, ["ward", "complete", "average", "single"]):
        plt.plot(*zip(*s), label=m)
    plt.legend()
    plt.ylabel("Score de Davies-Bouldin")
    plt.xlabel("Nombre de clusters")
    plt.title("Score de Davies-Bouldin en fonction du nombre de cluster et de la méthode")
    plt.savefig(f"agglomerative_clustering/{dataset}-davies_bouldin.png", dpi=100)
    plt.clf()
    for (s, m) in zip(calinski_harabaszs, ["ward", "complete", "average", "single"]):
        plt.plot(*zip(*s), label=m)
    plt.legend()
    plt.ylabel("Score de Calinski-Harabasz")
    plt.xlabel("Nombre de clusters")
    plt.title("Score de Calinski-Harabasz en fonction du nombre de cluster et de la méthode")
    plt.savefig(f"agglomerative_clustering/{dataset}-calinski_harabasz.png", dpi=100)
    plt.clf()
    for (s, m) in zip(silhouettes, ["ward", "complete", "average", "single"]):
        plt.plot(*zip(*s), label=m)
    plt.legend()
    plt.ylabel("Score de Silhouette")
    plt.xlabel("Nombre de clusters")
    plt.title("Score de Silhouette en fonction du nombre de cluster et de la méthode")
    plt.savefig(f"agglomerative_clustering/{dataset}-silhouette.png", dpi=100)
    plt.clf()
    return durations


def compute2(dataset, thresholds):
    print(f"dataset: {dataset}, {thresholds}")

    davies_bouldins = []
    calinski_harabaszs = []
    silhouettes = []
    durations = []
    cluster_counts = []
    for method in ["ward", "complete", "average", "single"]:
        davies_bouldin = []
        calinski_harabasz = []
        silhouette = []
        duration = []
        cluster_count = []
        for k in thresholds:
            t1 = time()
            data, agglomerative_clustering = plot2(dataset, k, method)
            cluster_count.append((k,agglomerative_clustering.n_clusters_))
            t2 = time()
            try:
                davies_bouldin.append((k, metrics.davies_bouldin_score(data, agglomerative_clustering.labels_)))
                calinski_harabasz.append((k, metrics.calinski_harabasz_score(data, agglomerative_clustering.labels_)))
                silhouette.append((k, metrics.silhouette_score(data, agglomerative_clustering.labels_)))
                duration.append((k, t2 - t1))
            except ValueError:
                print("err:", dataset, method, "d=", k)
        davies_bouldins.append(davies_bouldin)
        calinski_harabaszs.append(calinski_harabasz)
        silhouettes.append(silhouette)
        cluster_counts.append(cluster_count)
    for (s, m) in zip(davies_bouldins, ["ward", "complete", "average", "single"]):
        plt.plot(*zip(*s), label=m, marker='.')
    plt.legend()
    plt.ylabel("Score de Davies-Bouldin")
    plt.xlabel("Threshold")
    plt.title("Score de Davies-Bouldin en fonction du threshold et de la méthode")
    plt.savefig(f"agglomerative_clustering/{dataset}-threshold-davies_bouldin.png", dpi=100)
    plt.clf()
    for (s, m) in zip(cluster_counts, ["ward", "complete", "average", "single"]):
        plt.plot(*zip(*s), label=m, marker='.')
    plt.legend()
    plt.ylabel("Nombre de clusters")
    plt.xlabel("Threshold")
    plt.yscale('log')
    plt.title("Score de Davies-Bouldin en fonction du threshold et de la méthode")
    plt.savefig(f"agglomerative_clustering/{dataset}-cluster_count.png", dpi=100)
    plt.clf()
    plt.yscale('linear')
    for (s, m) in zip(calinski_harabaszs, ["ward", "complete", "average", "single"]):
        plt.plot(*zip(*s), label=m, marker='.')
    plt.legend()
    plt.ylabel("Score de Calinski-Harabasz")
    plt.xlabel("Threshold")
    plt.title("Score de Calinski-Harabasz en fonction du threshold et de la méthode")
    plt.savefig(f"agglomerative_clustering/{dataset}-threshold-calinski_harabasz.png", dpi=100)
    plt.clf()
    for (s, m) in zip(silhouettes, ["ward", "complete", "average", "single"]):
        plt.plot(*zip(*s), label=m, marker='.')
    plt.legend()
    plt.ylabel("Score de Silhouette")
    plt.xlabel("Threshold")
    plt.title("Score de Silhouette en fonction du threshold et de la méthode")
    plt.savefig(f"agglomerative_clustering/{dataset}-threshold-silhouette.png", dpi=100)
    plt.clf()
    return durations


times = []
models = ["xclara", "cassini", "3-spiral", ]
for d in models:
    times.append(compute(d))
for d, h in [("xclara", np.linspace(2, 102, 10)), ("cassini", np.linspace(0.1, 5.0, 100)), ("3-spiral", np.linspace(2, 20, 10))]:
    times.append(compute2(d, h))

for (duration, model) in zip(times, models):
    plt.plot(*zip(*duration), label=model)
plt.xlabel("Nombre de cluster")
plt.ylabel("Temps de calcul (en s)")
plt.title("Temps de calcul pour chaque modèle en fonction du nombre de cluster")
plt.legend()
plt.savefig(f"agglomerative_clustering/time.png", dpi=100)
plt.clf()

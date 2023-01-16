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

print()
warnings.filterwarnings("ignore")

path = './clustering-benchmark/src/main/resources/datasets/artificial/'

import scipy.cluster.hierarchy as shc

def load_data(dataset: str) -> np.ndarray:
    databrut = arff.loadarff(open(path + dataset + ".arff", 'r'))
    return np.array([[x[0], x[1]] for x in databrut[0]])


def compute(dataset, datanp, eps, min_samples, trace_time=False):
    begin_fit_time = time()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(datanp)
    labels = model.labels_
    end_fit_time = time()
    if trace_time:
        print(f"Fit time for {dataset}: {end_fit_time - begin_fit_time:.2}s")
    #
    begin_score_davies_bouldin = time()
    try:
        davies_bouldin_score = metrics.davies_bouldin_score(datanp, labels)
    except ValueError:
            davies_bouldin_score = 0
    end_score_davies_bouldin = time()
    if trace_time:
        print(f"Davies-Bouldin time for {dataset}: {end_score_davies_bouldin - begin_score_davies_bouldin:.2}s")

    begin_score_calinski_harabasz = time()
    try:
        calinski_harabasz_score = metrics.calinski_harabasz_score(datanp, labels)
    except ValueError:
        calinski_harabasz_score = 0
    end_score_calinski_harabasz = time()
    if trace_time:
        print(
            f"Calinski-Harabasz time for {dataset}): {end_score_calinski_harabasz - begin_score_calinski_harabasz:.2}s")
    begin_score_silhouette_score = time()
    try:
        silhouette_score = metrics.silhouette_score(datanp, labels)
    except ValueError:
        silhouette_score = 0
    end_score_silhouette_score = time()
    if trace_time:
        print(
            f"Silhouette time for {dataset}): {end_score_silhouette_score - begin_score_silhouette_score:.2}s")

    # print(f"Davies-Bouldin score for {dataset}: {davies_bouldin_score}")
    # print(f"Calinski-Harabasz score for {dataset}: {calinski_harabasz_score}")
    # print(f"Silhouette score for {dataset}: {calinski_harabasz_score}")

    f0 = datanp[:, 0]
    f1 = datanp[:, 1]
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(f"{dataset}: dbscan")
    os.makedirs(f"db_scan/{dataset}/", exist_ok=True)
    plt.savefig(f"db_scan/{dataset}/{eps}-{min_samples}.png")
    plt.clf()

    return davies_bouldin_score, calinski_harabasz_score, silhouette_score,


def generate_plot(dataset, trace_time=False, max_clusters=10):
    begin_load_time = time()
    datanp = load_data(dataset)
    end_load_time = time()
    if trace_time:
        print(f"Load time for {dataset}: {end_load_time - begin_load_time:.2}s")

    cluster_count = list(range(2, max_clusters))
    db_all = []
    ch_all = []
    s_all = []
    for eps in arange(0.01, 0.21, 0.05):
        db = []
        ch = []
        s = []
        print(eps)
        for min_samples in range(1, 101, 10):
            db_s, ch_s, s_s = compute(dataset, datanp, eps, min_samples, trace_time=trace_time)
            db.append(db_s)
            ch.append(ch_s)
            s.append(s_s)
        db_all.append(db)
        ch_all.append(ch)
        s_all.append(s)

    fig, ax = plt.subplots()
    im = plt.imshow(db_all, cmap='hot', interpolation='nearest')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Davies-Bouldin value", rotation=-90, va="bottom")
    plt.savefig(f"db_scan/{dataset}-db-score.png")
    plt.show()
    fig, ax = plt.subplots()
    im = plt.imshow(ch_all, cmap='hot', interpolation='nearest')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Calinski-Harabasz value", rotation=-90, va="bottom")
    plt.savefig(f"db_scan/{dataset}-ch-score.png")
    plt.show()
    fig, ax = plt.subplots()
    im = plt.imshow(ch_all, cmap='hot', interpolation='nearest')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Calinski-Harabasz value", rotation=-90, va="bottom")
    plt.savefig(f"db_scan/{dataset}-s-score.png")
    plt.show()


models = ["xclara", "cassini", "3-spiral", "golfball", "birch-rg1-score"]

for dataset in models:
    generate_plot(dataset)


# # Distances k plus proches voisins
# # Donnees dans X
# k = 5
# neigh = NearestNeighbors ( n_neighbors = k )
# neigh . fit ( X )
# distances , indices = neigh . kneighbors ( X )
# # retirer le point " origine "
# newDistances = np . asarray ( [ np . average ( distances [ i ] [ 1 : ] ) for i in range (0 ,
# distances . shape [ 0 ] ) ] )
# trie = np . sort ( newDistances )
# plt . title ( " Plus proches voisins ( 5 ) " )
# plt . plot ( trie ) ;
# plt . show ()
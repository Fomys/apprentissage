import os
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange
from scipy.io import arff
from sklearn import cluster, metrics
from time import time
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

print()
warnings.filterwarnings("ignore")

path = './clustering-benchmark/src/main/resources/datasets/artificial/'
path2 = './dataset-rapport/'

import scipy.cluster.hierarchy as shc

def load_data(dataset: str) -> np.ndarray:
    databrut = arff.loadarff(open(path + dataset + ".arff", 'r'))
    return np.array([[x[0], x[1]] for x in databrut[0]])

def load_data2(dataset : str) -> np.ndarray:
    databrut = np.loadtxt(path2 + dataset + ".txt")
    data = [[x[0],x[1]] for x in databrut]
    return np.array([[f[0], f[1]] for f in data])


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


def generate_plot(dataset, trace_time=False, max_clusters=20):
    begin_load_time = time()
    datanp = load_data2(dataset)
    end_load_time = time()
    if trace_time:
        print(f"Load time for {dataset}: {end_load_time - begin_load_time:.2}s")

    cluster_count = list(range(2, max_clusters))
    db_all = []
    ch_all = []
    s_all = []

    y = [1,10,20,30,80,100,200,250,300,350]
    x = [50000,65000]
    for eps in x:
        db = []
        ch = []
        s = []
        print(eps)
        for min_samples in y:
            db_s, ch_s, s_s = compute(dataset, datanp, eps, min_samples, trace_time=trace_time)
            db.append(db_s)
            ch.append(ch_s)
            s.append(s_s)
        db_all.append(db)
        ch_all.append(ch)
        s_all.append(s)




    # Distances k plus proches voisins
    # Donnees dans X
    k = 5
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(datanp)
    distances, indices = neigh.kneighbors(datanp)
    # retirer le point " origine "
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0, distances.shape[0])])
    trie = np.sort(newDistances)
    plt.title(dataset + ": Moyenne de la distance aux 5 plus proches voisins")
    plt.plot(trie);
    plt.savefig(f"db_scan/{dataset}-db-nearest.png")
    plt.show()





models = ["zz1","zz2"]

for dataset in models:
    generate_plot(dataset)




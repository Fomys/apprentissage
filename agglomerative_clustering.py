import os
import warnings

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster, metrics
from time import time

print()
warnings.filterwarnings("ignore")

path = './clustering-benchmark/src/main/resources/datasets/artificial/'

import scipy.cluster.hierarchy as shc


# Donnees dans datanp


# tps1 = time.time()
# model = cluster.AgglomerativeClustering(distance_threshold=10, linkage=' single ', n_clusters=None)
# model = model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_
# k = model.n_clusters_
# leaves = model.n_leaves_
# # Affichage clustering
# plt.scatter(f0, f1, c=labels, s=8)
# plt.title(" Resultat du clustering ")
# plt.show()
# print(" nb clusters = ", k, " , nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1) * 1000, 2), " ms ")
# # set the number of clusters
# k = 4
# tps1 = time.time()
# model = cluster.AgglomerativeClustering(linkage=' single ', n_clusters=k)
# model = model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_
# kres = model.n_clusters_
# leaves = model.n_leaves_


# matplotlib.use('Qt5Agg')

def load_data(dataset: str) -> np.ndarray:
    databrut = arff.loadarff(open(path + dataset + ".arff", 'r'))
    return np.array([[x[0], x[1]] for x in databrut[0]])


def compute(dataset, datanp, linkage, k, distance_threshold=None, trace_time=False):
    # print(" Dendrogramme 'single' donnees initiales ")
    # linked_mat = shc.linkage(datanp, 'single')
    # plt.figure(figsize=(12, 12))
    # shc.dendrogram(linked_mat,
    #                orientation='top',
    #                distance_sort='descending',
    #                show_leaf_counts=False)
    # plt.show()
    begin_fit_time = time()
    model = cluster.AgglomerativeClustering(linkage=linkage, n_clusters=k if distance_threshold is None else None, distance_threshold=distance_threshold, compute_full_tree=True if distance_threshold is None else 'auto')
    model.fit(datanp)
    labels = model.labels_
    end_fit_time = time()
    if trace_time:
        print(f"Fit time for {dataset} (k={k}): {end_fit_time - begin_fit_time:.2}s")
    #
    begin_score_davies_bouldin = time()
    davies_bouldin_score = metrics.davies_bouldin_score(datanp, labels)
    end_score_davies_bouldin = time()
    if trace_time:
        print(f"Davies-Bouldin time for {dataset} (k={k}: {end_score_davies_bouldin - begin_score_davies_bouldin:.2}s")

    begin_score_calinski_harabasz = time()
    calinski_harabasz_score = metrics.calinski_harabasz_score(datanp, labels)
    end_score_calinski_harabasz = time()
    if trace_time:
        print(
            f"Calinski-Harabasz time for {dataset} (k={k}): {end_score_calinski_harabasz - begin_score_calinski_harabasz:.2}s")
    begin_score_silhouette_score = time()
    silhouette_score = metrics.silhouette_score(datanp, labels)
    end_score_silhouette_score = time()
    if trace_time:
        print(
            f"Silhouette time for {dataset} (k={k}): {end_score_silhouette_score - begin_score_silhouette_score:.2}s")

    # print(f"Davies-Bouldin score for {dataset} (k={k}): {davies_bouldin_score}")
    # print(f"Calinski-Harabasz score for {dataset} (k={k}): {calinski_harabasz_score}")
    # print(f"Silhouette score for {dataset} (k={k}): {calinski_harabasz_score}")

    f0 = datanp[:, 0]
    f1 = datanp[:, 1]
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(f"{dataset}: agglomerative clustering with {k} clusters")
    os.makedirs(f"agglomerative_clustering/{dataset}", exist_ok=True)
    plt.savefig(f"agglomerative_clustering/{dataset}/k={k}-{linkage}.png")
    plt.clf()

    return davies_bouldin_score, calinski_harabasz_score, silhouette_score


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
    for linkage in ('ward', 'complete', 'average', 'single'):
        for distance_threshold in (None, 0.1, 0.5, 2.0):
            db = []
            ch = []
            s = []
            objects = []
            for k in cluster_count:
                db_s, ch_s, s_s = compute(dataset, datanp, linkage, k, trace_time=trace_time, distance_threshold=distance_threshold)
                db.append(db_s)
                ch.append(1 / ch_s)
                s.append(s_s)
                objects.append(f"k={k}")
            db_all.append(db)
            ch_all.append(ch)
            s_all.append(s)

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            db_plt = ax1.plot(cluster_count, db, color="r", label="Davies-Bouldin score")
            s_plt = ax1.plot(cluster_count, s, color="g", label="Silouhette score")
            ch_plt = ax2.plot(cluster_count, ch, color="b", label="Calinski-Harabasz (inverse)")
            # added these three lines
            lns = db_plt + ch_plt + s_plt
            labs = [l.get_label() for l in lns]
            plt.legend(lns, labs)
            plt.title(f"{dataset}: scores vs number of clusters with linkage {linkage}\ndistance threshold of {distance_threshold}")
            plt.savefig(f"agglomerative_clustering/{dataset}-{linkage}-{distance_threshold}-score.png")
            plt.show()



models = ["xclara", "cassini", "3-spiral", "golfball"]

for dataset in models:
    generate_plot(dataset)

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster, metrics
from time import time
print()
warnings.filterwarnings("ignore")

path = './clustering-benchmark/src/main/resources/datasets/artificial/'
path2 = './dataset-rapport/'


def load_data(dataset: str) -> np.ndarray:
    databrut = arff.loadarff(open(path + dataset + ".arff", 'r'))
    return np.array([[x[0], x[1]] for x in databrut[0]])



def load_data2(dataset : str) -> np.ndarray:
    databrut = np.loadtxt(path2 + dataset + ".txt")
    data = [[x[0],x[1]] for x in databrut]
    return np.array([[f[0], f[1]] for f in data])

def compute(dataset, datanp, k, trace_time=False):
    begin_fit_time = time()
    model = cluster.KMeans(n_clusters=k, init="k-means++")
    model.fit(datanp)
    labels = model.labels_
    end_fit_time = time()
    if trace_time:
        print(f"Fit time for {dataset} (k={k}): {end_fit_time - begin_fit_time:.2}s")

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

    f0 = datanp[:, 0]
    f1 = datanp[:, 1]
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(f"{dataset}: k-means with {k} clusters")
    plt.savefig(f"k-means/{dataset}-k={k}.png")
    plt.clf()

    return davies_bouldin_score, calinski_harabasz_score, silhouette_score, end_fit_time - begin_fit_time


def generate_plot(dataset, trace_time=False, max_clusters=3):
    begin_load_time = time()
    datanp = load_data2(dataset)
    end_load_time = time()
    if trace_time:
        print(f"Load time for {dataset}: {end_load_time - begin_load_time:.2}s")
    db = []
    ch = []
    s = []
    t = []
    cluster_count = list(range(2, max_clusters))
    for k in cluster_count:
        print(k)
        db_s, ch_s, s_s, t_ = compute(dataset, datanp, k, trace_time=trace_time)
        db.append(db_s)
        ch.append(1/ch_s)
        s.append(s_s)
        t.append(t_)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    db_plt = ax1.plot(cluster_count, db, color="r", label="Davies-Bouldin")
    s_plt = ax1.plot(cluster_count, s, color="g", label="Silhouette")
    ch_plt = ax2.plot(cluster_count, ch, color="b", label="Calinski-Harabasz (inverse)")
    # added these three lines
    lns = db_plt + ch_plt + s_plt
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs)
    plt.title(f"{dataset}: scores vs number of clusters")
    plt.savefig(f"k-means/{dataset}-score.png")
    plt.show()
    print(t)
    plt.plot(cluster_count, t, color="r", label="Compute time")
    plt.legend()
    plt.title(f"{dataset}: Compute time for k-means method")
    plt.savefig(f"k-means/{dataset}-compute-time.png")
    plt.show()


models = ["y1"]

for dataset in models:
    print(dataset)
    generate_plot(dataset)




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

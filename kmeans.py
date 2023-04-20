from math import inf, sqrt
from random import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def dist(p1: List[float], p2: List[float]):
    return sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(p1, p2)]))


def remap(val: float, old_min: float, old_max: float, min: float, max: float):
    return ((val - old_min) / (old_max - old_min)) * max + min


def init_centroid(
    k: int, dim: int, min: float = 0, max: float = 1
) -> List[List[float]]:
    rand_tuple = lambda: [remap(random(), 0, 1, min, max) for _ in range(0, dim)]
    return [rand_tuple() for _ in range(0, k)]


def cluster(centroids: List[List[float]], data: List[List[float]]) -> Dict[int, int]:
    assert len(centroids) > 0
    assert len(data) > 0
    clusters = {index: [] for index in range(0, len(centroids))}

    def cluster_point(point: List[float]) -> int:
        min_dist = inf
        min_index = -1
        for id, centroid in enumerate(centroids):
            d = dist(point, centroid)
            if d < min_dist:
                min_dist = d
                min_index = id
        return min_index

    for index, point in enumerate(data):
        clusters[cluster_point(point)].append(index)
    return clusters


def update_centroids(
    clusters: Dict[int, int], data: List[List[float]]
) -> List[List[float]]:
    ret = [[] for _ in range(0, len(clusters.keys()))]
    for cluster in clusters.keys():
        coords = [0 for _ in range(0, len(data[0]))]  # Get the number of elements
        for datapoint in clusters[cluster]:
            coords = [
                coords[i] + data[datapoint][i] for i in range(0, len(data[datapoint]))
            ]
        coords = [coords[i] / len(clusters[cluster]) for i in range(0, len(coords))]
        ret[cluster] = coords
    return ret


def error(
    clusters: Dict[int, int], data: List[List[float]], centroids: List[List[float]]
) -> float:
    ret = 0
    for cluster in clusters.keys():
        for datapoint in clusters[cluster]:
            ret += dist(data[datapoint], centroids[cluster]) ** 2
    return ret


def km(data: List[List[float]], k: int):
    x, y = len(data), len(data[0])
    min_val, max_val = inf, -inf
    for i in range(0, x):
        for j in range(0, y):
            if data[i][j] < min_val:
                min_val = data[i][j]
            if data[i][j] > max_val:
                max_val = data[i][j]
    centroids = init_centroid(k, 1, min=min_val, max=max_val)
    clusters = cluster(centroids, data)

    def update_until_converge(clusters):
        new_centroids = update_centroids(clusters, data)
        new_clusters = cluster(new_centroids, data)
        if all(new_clusters) == all(clusters):
            return new_clusters, new_centroids
        else:
            return update_until_converge(new_clusters)

    return update_until_converge(clusters)


def kmautomatic(data: List[List[float]]):
    k = 2
    errors = []

    def update_until_worsen(k):
        clusters, centroids = km(data, k)
        errors.append(error(clusters, data, centroids))

        if len(errors) > 1 and errors[-1] > errors[-2]:
            return -1
        else:
            ret = update_until_worsen(k + 1)
            if ret == -1:
                return clusters, centroids, k
            else:
                return ret

    clusters, centroids, k = update_until_worsen(k)
    return clusters, centroids, k, errors


def plot_2d(clusters, centroids, k, errors):
    # Plot the errors
    plt.figure("Error as a function of k")
    plt.plot(range(2, k + 2), errors)
    plt.scatter(k, errors[k - 2], color="red", marker="x", s=100)

    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])

    # Color each cluster with a different color
    plt.figure("Clusters")
    colors = ["red", "green", "blue", "yellow", "orange"]

    for i in range(0, len(clusters)):
        x = np.array([data[j][0] for j in clusters[i]])
        y = np.array([data[j][1] for j in clusters[i]])
        if i < len(colors):
            plt.scatter(x, y, color=colors[i])

    # Plot the centroids
    x = np.array([i[0] for i in centroids])
    y = np.array([i[1] for i in centroids])
    for i in range(0, len(centroids)):
        if i < len(colors):
            plt.scatter(x[i], y[i], color=colors[i], marker="x", s=100)
        else:
            plt.scatter(x[i], y[i], color="black", marker="x", s=100)
    plt.show()


if __name__ == "__main__":
    data = init_centroid(100, 2, min=100, max=10000)

    clusters, centroids, k, errors = kmautomatic(data)
    plot_2d(clusters, centroids, k, errors)

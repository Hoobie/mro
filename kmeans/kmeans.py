import cPickle
import gzip
import random

import numpy
import matplotlib.pyplot as plt
from sklearn import decomposition

MAX_ITER = 30
THRESHOLD = 150


def select_random_seeds(x, k):
    s = []
    for i in xrange(0, k):
        rand = random.randint(0, len(x) - 1)
        s.append(x[rand])
    return s


def rss(s, u, k):
    wcss = 0
    for i in xrange(0, k):
        for x in s[i]:
            wcss += numpy.linalg.norm(x - u[i])
    return wcss


def k_means(x, k):
    u = select_random_seeds(x, k)
    iteration = 0
    rss_values = []
    sets = []
    while iteration < MAX_ITER:
        s = None
        for i in xrange(0, k):
            w = [[] for _ in xrange(0, k)]
            for x_i in x:
                distances = []
                for u_i in u:
                    distances.append(numpy.linalg.norm(u_i - x_i))
                min_idx = numpy.argmin(distances)
                w[min_idx].append(x_i)
            for j in xrange(0, k):
                if len(w[j]) > 0:
                    u[j] = numpy.mean(w[j])
            s = w
        rss_value = rss(s, u, k)
        rss_values.append(rss_value)
        sets.append(s)
        if rss_value < THRESHOLD:
            break
        iteration += 1
    return u, sets, rss_values


def plot_rss():
    # f = gzip.open('../noise.pkl.gz', 'rb')
    # data_set = cPickle.load(f)
    f = gzip.open('../mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    test_x, test_y = test_set
    f.close()
    k_means_results = []
    rss_results = []
    k_range = range(1, 20)
    for i in k_range:
        # k_means_result = k_means(data_set, i)
        k_means_result = k_means(test_x[0:1000], i)
        k_means_results.append(k_means_result[0])
        rss_results.append(k_means_result[2][len(k_means_result[2]) - 1])
    plt.plot(k_range, rss_results, label="RSS")
    plt.show()


def visualize_noise():
    f = gzip.open('../noise.pkl.gz', 'rb')
    data_set = cPickle.load(f)
    f.close()

    noise_results = k_means(data_set, 3)[1]
    noise_sets = noise_results[len(noise_results) - 1]

    pca = decomposition.PCA(n_components=2)
    pca.fit(data_set)

    transformed = pca.transform(noise_sets[0])
    plt.scatter(transformed[:, 0], transformed[:, 1], color='red')
    transformed = pca.transform(noise_sets[1])
    plt.scatter(transformed[:, 0], transformed[:, 1], color='blue')
    transformed = pca.transform(noise_sets[2])
    plt.scatter(transformed[:, 0], transformed[:, 1], color='green')
    plt.show()


if __name__ == "__main__":
    # plot_rss()
    visualize_noise()

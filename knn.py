import copy
import gzip
import cPickle
import sys
import random
import time
from collections import Counter

import matplotlib.pyplot as plt

import numpy

SAMPLE_COUNT = 1000


def count_distance(vector1, vector2):
    dist = 0
    for x1, x2 in zip(vector1, vector2):
        dist += pow(x2 - x1, 2)
    return dist


def implementation1_1nn():
    error_count = 0
    for test_vector, test_digit in zip(test_x_copy, test_y):
        nearest = (sys.maxint, [], [])
        for train_vector, train_digit in zip(train_x_copy, train_y):
            distance = count_distance(test_vector, train_vector)
            if distance < nearest[0]:
                nearest = (distance, test_vector, train_digit)
        if nearest[2] != test_digit:
            error_count += 1
    return error_count


def implementation2_1nn():
    error_count = 0
    for test_vector, test_digit in zip(test_x_copy, test_y):
        subtracting = train_x_copy - test_vector
        powered = subtracting ** 2
        summed = numpy.sum(powered, axis=1)
        min_index = numpy.argmin(summed)
        if test_digit != train_y[min_index]:
            error_count += 1
    return error_count


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def knn(k):
    error_count = 0
    for test_vector, test_digit in zip(test_x_copy, test_y):
        subtracting = train_x_copy - test_vector
        powered = subtracting ** 2
        summed = numpy.sum(powered, axis=1)
        neighbours = []
        for i in xrange(1, k + 1):
            min_index = numpy.argmin(summed)
            neighbours.append(train_y[min_index])
            summed[min_index] = sys.maxint
        mc = most_common(neighbours)
        if test_digit != mc:
            error_count += 1
    return error_count


def nearest_mean():
    x_vectors = [[] for _ in xrange(10)]
    for train_vector, train_digit in zip(train_x_copy, train_y):
        x_vectors[train_digit].append(train_vector)
    means = [None for _ in xrange(10)]
    for i in xrange(10):
        means[i] = (numpy.mean(x_vectors[i], axis=0))

    error_count = 0
    for test_vector, test_digit in zip(test_x_copy, test_y):
        nearest = (sys.maxint, [], [])
        for i in xrange(10):
            distance = count_distance(test_vector, means[i])
            if distance < nearest[0]:
                nearest = (distance, test_vector, i)
        if test_digit != nearest[2]:
            error_count += 1
    return error_count


def random_vector_scale(vector, scale):
    index = random.randint(0, 728)
    for x in vector:
        x[index] *= scale
    return vector


if __name__ == "__main__":
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = train_y[0:SAMPLE_COUNT]
    test_y = test_y[0:SAMPLE_COUNT]

    errors = []

    # s_range = [1]
    # s_range = xrange(0, 21)
    # for s in s_range:
    # k_range = xrange(1, 20)
    # for k_param in k_range:
    train_x_copy = random_vector_scale(copy.deepcopy(train_x[0:SAMPLE_COUNT]), 1)
    test_x_copy = random_vector_scale(copy.deepcopy(test_x[0:SAMPLE_COUNT]), 1)

    start_time = time.time()

    # print "\nk:", k_param

    error_percentage = nearest_mean() * 100 / SAMPLE_COUNT
    errors.append(error_percentage)
    print "Error percentage:", error_percentage, "%"

    elapsed_time = time.time() - start_time
    print "Elapsed time:", elapsed_time, "s"

    # plt.plot(k_range, errors, label="error percentage")
    # plt.xlabel("k")
    # plt.ylabel("errors [%]")
    # plt.legend()
    # plt.show()

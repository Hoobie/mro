import copy
import gzip
import cPickle
import sys
import random
import time

import matplotlib.pyplot as plt

import numpy

SAMPLE_COUNT = 500


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
    s_range = xrange(0, 21)
    for s in s_range:
        train_x_copy = random_vector_scale(copy.deepcopy(train_x[0:SAMPLE_COUNT]), s)
        test_x_copy = random_vector_scale(copy.deepcopy(test_x[0:SAMPLE_COUNT]), s)

        start_time = time.time()

        print "\ns:", s

        error_percentage = implementation2_1nn() * 100 / SAMPLE_COUNT
        errors.append(error_percentage)
        print "Error percentage:", error_percentage, "%"

        elapsed_time = time.time() - start_time
        print "Elapsed time:", elapsed_time, "s"

    plt.plot(s_range, errors, label="error percentage")
    plt.xlabel("s")
    plt.ylabel("errors [%]")
    plt.legend()
    plt.show()

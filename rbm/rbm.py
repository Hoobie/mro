import cPickle
import gzip
import random

import numpy
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM


def normalize(data_set):
    return (data_set - numpy.min(data_set, 0)) / (numpy.max(data_set, 0) + 0.0001)


def visualize_filters(rbm, n_components, x, y):
    plt.clf()
    for i, comp in enumerate(rbm.components_):
        plt.subplot(x, y, i + 1)
        plt.imshow(comp.reshape((28, 28)), cmap='gray', interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        plt.suptitle('%s filters extracted by RBM' % n_components, fontsize=10)
    plt.savefig('filters_%s.png' % n_components)


def visualize_sampling(rmb, sample, steps, filename):
    for i in xrange(0, steps):
        new_sample = rmb.gibbs(sample)
        plt.imshow(new_sample.reshape(28, 28), cmap='gray')
        plt.savefig(filename + '_' + str(i + 1) + '.png')


if __name__ == "__main__":
    f = gzip.open('../mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    train_x, train_y = train_set
    train_x = train_x[0:2000]
    test_x, test_y = test_set
    test_x = test_x[0:2000]
    f.close()

    # 1

    normalized_train_x = normalize(train_x)
    normalized_test_x = normalize(test_x)

    rbm20 = BernoulliRBM(batch_size=100, learning_rate=0.1, n_components=20, n_iter=20, random_state=None, verbose=1)
    rbm20.fit(normalized_train_x)
    visualize_filters(rbm20, 20, 4, 5)

    rbm100 = BernoulliRBM(batch_size=100, learning_rate=0.1, n_components=100, n_iter=20, random_state=None, verbose=1)
    rbm100.fit(normalized_train_x)
    visualize_filters(rbm100, 100, 10, 10)

    # 2
    random_set = [random.uniform(0.0, 1.0) for _ in xrange(0, 784)]

    visualize_sampling(rbm20, normalized_test_x[0], 5, 'rbm20_test1')
    visualize_sampling(rbm20, normalized_test_x[1], 5, 'rbm20_test2')
    visualize_sampling(rbm20, random_set, 5, 'rbm20_noise')

    visualize_sampling(rbm100, normalized_test_x[0], 5, 'rbm100_test1')
    visualize_sampling(rbm100, normalized_test_x[1], 5, 'rbm100_test2')
    visualize_sampling(rbm100, random_set, 5, 'rbm100_noise')

import cPickle
import gzip

from sklearn import decomposition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_components_to_file(data, n, file_name):
    pca = decomposition.PCA(n_components=n)
    pca.fit(data)
    for i in xrange(0, n):
        component = pca.components_[i].reshape(28, 28)
        plt.imshow(component, cmap='gray')
        plt.savefig(file_name + str(i + 1) + '.png')


if __name__ == "__main__":
    f = gzip.open('../mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    train_x, train_y = train_set
    train_x = train_x[0:2000]
    test_x, test_y = test_set
    test_x = test_x[0:2000]
    f.close()

    # 1
    plot_components_to_file(train_x, 2, 'ex1_')

    # 2
    plot_components_to_file(train_x, 5, 'ex2_')

    # 3
    plt.imshow(test_x[0].reshape(28, 28), cmap='gray')
    plt.savefig('ex3_orig1.png')
    plt.imshow(test_x[1].reshape(28, 28), cmap='gray')
    plt.savefig('ex3_orig2.png')
    for i in [5, 10, 15, 20, 50]:
        pca = decomposition.PCA(n_components=i)
        pca.fit(test_x)
        transformed = pca.transform(test_x[0])
        inversed = pca.inverse_transform(transformed).reshape(28, 28)

        plt.imshow(inversed, cmap='gray')
        plt.savefig('ex3_1_' + str(i) + '.png')

        pca = decomposition.PCA(n_components=i)
        pca.fit(test_x)
        transformed = pca.transform(test_x[1])
        inversed = pca.inverse_transform(transformed).reshape(28, 28)

        plt.imshow(inversed, cmap='gray')
        plt.savefig('ex3_2_' + str(i) + '.png')

    # 4
    model = TSNE(n_components=2, random_state=0)
    model = model.fit_transform(test_x)
    plt.plot(*zip(*model), marker='o', color='b', ls='')
    plt.savefig('ex4_tsne.png')

    pca = decomposition.PCA(n_components=2)
    pca.fit(test_x)
    transformed = pca.transform(test_x)
    plt.plot(*zip(*transformed), marker='o', color='r', ls='')
    plt.savefig('ex4_pca.png')

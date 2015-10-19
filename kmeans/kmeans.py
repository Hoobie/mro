import cPickle
import gzip



if __name__ == "__main__":
    f = gzip.open('../noise.pkl.gz', 'rb')
    data_set = cPickle.load(f)
    f.close()


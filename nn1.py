import gzip
import cPickle
import sys


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
test_x, test_y = test_set

for test_vector, test_digit in zip(test_x, test_y)[0:50]:
    nearest = (sys.maxint, [], [])
    for train_vector, train_digit in zip(train_x, train_y)[0:500]:
        dist = 0
        for test_xi, train_xi in zip(test_vector, train_vector):
            dist += pow(train_xi - test_xi, 2)
        if dist < nearest[0]:
            nearest = (dist, test_vector, train_digit)
    if nearest[2] == test_digit:
        print "OK"
    else:
        print "ERROR"




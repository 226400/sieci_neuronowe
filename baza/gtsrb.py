import cv2
import numpy as np

import csv
from matplotlib import cm
from matplotlib import pyplot as plt

def load_data(rootpath="datasets/gtsrb_training", feature=None, cut_roi=True,
              test_split=0.2, plot_samples=False, seed=113):
    classes = np.arange(0, 42, 2)

    X = []
    labels = []
    for c in xrange(len(classes)):
        prefix = rootpath + '/' + format(classes[c], '05d') + '/'

        gt_file = open(prefix + 'GT-' + format(classes[c], '05d') + '.csv')

        gt_reader = csv.reader(gt_file, delimiter=';')
        gt_reader.next()

        for row in gt_reader:
            im = cv2.imread(prefix + row[0])

            if cut_roi:
                im = im[np.int(row[4]):np.int(row[6]),
                        np.int(row[3]):np.int(row[5]), :]

            X.append(im)
            labels.append(c)
        gt_file.close()

    X = _extract_feature(X, feature)

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if plot_samples:
        num_samples = 15
        sample_idx = np.random.randint(len(X), size=num_samples)
        sp = 1
        for r in xrange(3):
            for c in xrange(5):
                ax = plt.subplot(3, 5, sp)
                sample = X[sample_idx[sp - 1]]
                ax.imshow(sample.reshape((32, 32)), cmap=cm.Greys_r)
                ax.axis('off')
                sp += 1
        plt.show()

    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]

    return (X_train, y_train), (X_test, y_test)


def _extract_feature(X, feature):

    if feature == 'gray' or feature == 'surf':
        X = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X]
    elif feature == 'hsv':
        X = [cv2.cvtColor(x, cv2.COLOR_BGR2HSV) for x in X]

    small_size = (32, 32)
    X = [cv2.resize(x, small_size) for x in X]

    if feature == 'surf':
        surf = cv2.SURF(400)
        surf.upright = True
        surf.extended = True
        num_surf_features = 36

        dense = cv2.FeatureDetector_create("Dense")
        kp = dense.detect(np.zeros(small_size).astype(np.uint8))

        kp_des = [surf.compute(x, kp) for x in X]

        X = [d[1][:num_surf_features, :] for d in kp_des]
    elif feature == 'hog':
        block_size = (small_size[0] / 2, small_size[1] / 2)
        block_stride = (small_size[0] / 4, small_size[1] / 4)
        cell_size = block_stride
        num_bins = 9
        hog = cv2.HOGDescriptor(small_size, block_size, block_stride,
                                cell_size, num_bins)
        X = [hog.compute(x) for x in X]
    elif feature is not None:
        X = np.array(X).astype(np.float32) / 255

        X = [x - np.mean(x) for x in X]

    X = [x.flatten() for x in X]
    return X

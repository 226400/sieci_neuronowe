#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np

from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as plt



class Classifier:
      This class also provides method to calculate accuracy, precision,
        recall, and the confusion matrix.
  

        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        # all cases where predicted class was correct
        mask = y_hat == y_test
        return np.float32(np.count_nonzero(mask)) / len(y_test)

    def _precision(self, y_test, Y_vote):
 
        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        if self.mode == "one-vs-one":
            # need confusion matrix
            conf = self._confusion(y_test, Y_vote)

            # consider each class separately
            prec = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = conf[c, c]

                # false positives: label is c, classifier predicted not c
                fp = np.sum(conf[:, c]) - conf[c, c]

                if tp + fp != 0:
                    prec[c] = tp * 1. / (tp + fp)
        elif self.mode == "one-vs-all":
            # consider each class separately
            prec = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = np.count_nonzero((y_test == c) * (y_hat == c))

                # false positives: label is c, classifier predicted not c
                fp = np.count_nonzero((y_test == c) * (y_hat != c))

                if tp + fp != 0:
                    prec[c] = tp * 1. / (tp + fp)
        return prec

    def _recall(self, y_test, Y_vote):
   
        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        if self.mode == "one-vs-one":
            # need confusion matrix
            conf = self._confusion(y_test, Y_vote)

            # consider each class separately
            recall = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = conf[c, c]

                # false negatives: label is not c, classifier predicted c
                fn = np.sum(conf[c, :]) - conf[c, c]
                if tp + fn != 0:
                    recall[c] = tp * 1. / (tp + fn)
        elif self.mode == "one-vs-all":
            # consider each class separately
            recall = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = np.count_nonzero((y_test == c) * (y_hat == c))

                # false negatives: label is not c, classifier predicted c
                fn = np.count_nonzero((y_test != c) * (y_hat == c))

                if tp + fn != 0:
                    recall[c] = tp * 1. / (tp + fn)
        return recall

    def _confusion(self, y_test, Y_vote):
    
        y_hat = np.argmax(Y_vote, axis=1)
        conf = np.zeros((self.num_classes, self.num_classes)).astype(np.int32)
        for c_true in xrange(self.num_classes):
            # looking at all samples of a given class, c_true
            # how many were classified as c_true? how many as others?
            for c_pred in xrange(self.num_classes):
                y_this = np.where((y_test == c_true) * (y_hat == c_pred))
                conf[c_pred, c_true] = np.count_nonzero(y_this)
        return conf


class MultiClassSVM(Classifier):
  

    def __init__(self, num_classes, mode="one-vs-all", params=None):
    
        self.num_classes = num_classes
        self.mode = mode
        self.params = params or dict()

        # initialize correct number of classifiers
        self.classifiers = []
        if mode == "one-vs-one":
            # k classes: need k*(k-1)/2 classifiers
            for _ in xrange(num_classes*(num_classes - 1) / 2):
                self.classifiers.append(cv2.SVM())
        elif mode == "one-vs-all":
            # k classes: need k classifiers
            for _ in xrange(num_classes):
                self.classifiers.append(cv2.SVM())
        else:
            print "Unknown mode ", mode

    def fit(self, X_train, y_train, params=None):
   
        if params is None:
            params = self.params

        if self.mode == "one-vs-one":
            svm_id = 0
            for c1 in xrange(self.num_classes):
                for c2 in xrange(c1 + 1, self.num_classes):
                    # indices where class labels are either `c1` or `c2`
                    data_id = np.where((y_train == c1) + (y_train == c2))[0]

                    # set class label to 1 where class is `c1`, else 0
                    y_train_bin = np.where(y_train[data_id] == c1, 1, 
                                           0).flatten()

                    self.classifiers[svm_id].train(X_train[data_id, :],
                                                   y_train_bin,
                                                   params=self.params)
                    svm_id += 1
        elif self.mode == "one-vs-all":
            for c in xrange(self.num_classes):
                # train c-th SVM on class c vs. all other classes
                # set class label to 1 where class==c, else 0
                y_train_bin = np.where(y_train == c, 1, 0).flatten()

                # train SVM
                self.classifiers[c].train(X_train, y_train_bin,
                                          params=self.params)

    def evaluate(self, X_test, y_test, visualize=False):
      
        Y_vote = np.zeros((len(y_test), self.num_classes))

        if self.mode == "one-vs-one":
            svm_id = 0
            for c1 in xrange(self.num_classes):
                for c2 in xrange(c1 + 1, self.num_classes):
                    data_id = np.where((y_test == c1) + (y_test == c2))[0]
                    X_test_id = X_test[data_id, :]
                    y_test_id = y_test[data_id]

                    
                    # predict labels
                    y_hat = self.classifiers[svm_id].predict_all(X_test_id)

                    for i in xrange(len(y_hat)):
                        if y_hat[i] == 1:
                            Y_vote[data_id[i], c1] += 1
                        elif y_hat[i] == 0:
                            Y_vote[data_id[i], c2] += 1
                        else:
                            print "y_hat[", i, "] = ", y_hat[i]

                  
                    svm_id += 1
        elif self.mode == "one-vs-all":
            for c in xrange(self.num_classes):
              
                y_hat = self.classifiers[c].predict_all(X_test)

                # we vote for c where y_hat is 1
                if np.any(y_hat):
                    Y_vote[np.where(y_hat == 1)[0], c] += 1

          
            no_label = np.where(np.sum(Y_vote, axis=1) == 0)[0]
            Y_vote[no_label, np.random.randint(self.num_classes,
                                               size=len(no_label))] = 1

        accuracy = self._accuracy(y_test, Y_vote)
        precision = self._precision(y_test, Y_vote)
        recall = self._recall(y_test, Y_vote)
        return accuracy, precision, recall

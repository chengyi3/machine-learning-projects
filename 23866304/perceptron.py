# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
# import timeit
import collections
from collections import Counter
# import math
def classify(train_set, train_labels, dev_set, learning_rate,max_iter):
    """
    train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
                This can be thought of as a list of 7500 vectors that are each
                3072 dimensional.  We have 3072 dimensions because there are
                each image is 32x32 and we have 3 color channels.
                So 32*32*3 = 3072
    train_labels - List of labels corresponding with images in train_set
    example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
             and X1 is a picture of a dog and X2 is a picture of an airplane.
             Then train_labels := [1,0] because X1 contains a picture of an animal
             and X2 contains no animals in the picture.

    dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
              It is the same format as train_set
    """
    # TODO: Write your code here
    w = np.zeros(len(train_set[0]))
    # print(len(train_set[0]))
    learning_rate = 0.5
    #classifier
    labels = []
    for i in range(0, len(train_labels)):
        if train_labels[i] == 0:
            labels.append(-1)
        else:
            labels.append(1)

    # print(len(labels))
    bias = 0
    for i in range(0, max_iter):
        np.random.seed(0)
        for j in np.random.permutation(len(train_set)):
            imagex = train_set[j]
            d = np.dot(w, imagex)
            f = d + bias
            if (f >= 0):
                sign = 1
            else:
                sign = -1
            if (sign != labels[j]):
                # for x in range(len(w)):
                    # w[x] = w[x] + (learning_rate*(labels[j]))*imagex[x]
                w = w + (learning_rate*(labels[j]))*imagex
                bias = bias + learning_rate*(labels[j] - sign)
    # print(w)
    predict_label = []
    for i in range(0, len(dev_set)):
        image = dev_set[i]
        d = np.dot(w, image)
        f = d+bias
        if f >= 0:
            predict_label.append(1)
        else:
            predict_label.append(0)

    # print(len(predict_label))



    # return predicted labels of development set
    return predict_label



def classifyEC(train_set, train_labels, dev_set,learning_rate,max_iter):
    # labels = []
    # for i in range(0, len(train_labels)):
    #     if train_labels[i] == 0:
    #         labels.append(0)
    #     else:
    #         labels.append(1)

    # start = timeit.default_timer()
    predict = []
    k =  7
    for i, image in enumerate(dev_set):
        # image = dev_set[i]
        distance = Counter()
        # distance = np.zeros(len(train_set))
        for j, imagej in enumerate(train_set):
            # print(imagej)
            # print(j)
            # distance = []

            # k = 10
            # w = image
            # w = w - imagej
            # sum = 0
            # # for m, y in np.ndenumerate(len(w)):
            #     sum = sum + abs(y)
            distance[j] = sum(abs(imagej - image))
            #
            # distance[j] = sum
            # if (distance[j[0]] == 0):
            # distance[j] = np.linalg.norm(image - imagej)
        dist = sorted(distance.items(), key=lambda x: x[1])
        # print(distance)
        label = []
        # index = np.argpartition(distance, k)

        # print(index)
        for o in range(k):
            n = dist[o][0]
            label.append(train_labels[n])
            # idx = index[o]
            # label.append(train_labels[idx])
            # print(label)

        predict_label = np.bincount(label).argmax()
        if (predict_label == True):
            predict.append(1)
        else:
            predict.append(0)
        # if (predict_label == 1):
        #     predict.append(1)
        # else:
        #     predict.append(0)
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)
    return predict
    # Write your code here if you would like to attempt the extra credit

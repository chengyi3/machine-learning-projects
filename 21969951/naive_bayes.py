# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
import nltk
import collections
from collections import Counter
# from datatime import datatime
# import timeit
STEMMING = True
LOWER_CASE = True

# start = timeit.default_timer()
def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)
    """
    # TODO: Write your code here
    # return predicted labels of development set
    # count_positiveword = {}

    smoothing_parameter = 0.01

    count_positiveword = Counter()
    positive_word = []
    # count_negtiveword = {}
    count_negtiveword = Counter()
    negative_word = []
    probabilitypos = Counter()
    probabilityneg = Counter()
    pre_dev_label = []
    a = 0
    b = 0
    for i in range(0, len(train_set)):
        if (train_labels[i] == 1):
            for word in train_set[i]:
                    count_positiveword[word] += 1
                    a = a+1
                    if (len(count_positiveword) == 2500):
                        break
                # if word not in count_positiveword:
                #     # print(positive_word)
                #     positive_word.append(word)
                #     count_positiveword[word] = 1
                # else:
                #     count_positiveword[word] += 1
        else:
            for word in train_set[i]:
                    count_negtiveword[word] += 1
                    b = b+1
                    if (len(count_negtiveword) == 2500):
                        break
                # if word not in count_negtiveword:
                #     negative_word.append(word)
                #     count_negtiveword[word] = 1
                # if word in count_negtiveword:
                #     count_negtiveword[word] += 1
    for word in count_positiveword:
        probabilitypos[word] = (smoothing_parameter + count_positiveword[word])/(a + smoothing_parameter*(2500 + 1))
        # a = a+1
        # b = b+count_positiveword[word]
        # if (a == 2500):
        #     break
    for word in count_negtiveword:
        probabilityneg[word] = (smoothing_parameter + count_negtiveword[word])/(b + smoothing_parameter*(2500 + 1))
    unseenpositivepro = smoothing_parameter/(a + smoothing_parameter*(2500 + 1))
    unseennegtivepro = smoothing_parameter/(b + smoothing_parameter*(2500 + 1))
    # for word in count_positiveword:
    #     probabilitypos[word] = (smoothing_parameter + count_positiveword[word])/(2500 + smoothing_parameter*(len(count_positiveword) + 1))
    #     # if (len(probabilitypos) == 2500):
    #     #     break
    # for word in count_negtiveword:
    #     probabilityneg[word] = (smoothing_parameter + count_negtiveword[word])/(2500 + smoothing_parameter*(len(count_negtiveword) + 1))
        # if (len(probabilityneg) == 2500):
        #     break
    for i in range(0, len(dev_set)):
        prob_positive = np.log(pos_prior)
        prob_negtive = np.log(1-pos_prior)
        # prob_positive = 0
        # prob_negtive = 0
        for word in dev_set[i]:
            if word in count_positiveword:
                prob_positive += np.log(probabilitypos[word])
            else:
                prob_positive += np.log(unseenpositivepro)
            if word in count_negtiveword:
                prob_negtive += np.log(probabilityneg[word])
            else:
                prob_negtive += np.log(unseennegtivepro)
        if prob_positive  >= prob_negtive:
            pre_dev_label.append(1)
        else:
        # elif prob_negtive + prob_negtive0 >= prob_positive + prob_positive0:
            pre_dev_label.append(0)
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)
    return pre_dev_label

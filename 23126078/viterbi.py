"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
import collections
# import timeit
from collections import Counter

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    predicts = []
    counttagofeachword = Counter()
    for sentence in train:
        for wordtag in sentence:
            word = wordtag[0]
            tag = wordtag[1]
            if word in counttagofeachword:
                counttagofeachword[word][tag] += 1
            else:
                counttagofeachword[word] = Counter()
                counttagofeachword[word][tag] += 1
    for sentence in test:
        predict = []
        for word in sentence:
            if word not in counttagofeachword:
                predict.append((word, 'NOUN'))
            else:
                tag = counttagofeachword[word].most_common(1)[0][0]
                predict.append((word, tag))
        predicts.append(predict)





    # raise Exception("You must implement me")
    return predicts


def viterbi(train, test):
    '''
    TODO: implement the Viterbi algorithm.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # start = timeit.default_timer()
    predicts = []
    initial_Tag = Counter()
    counttagofeachword = Counter()
    tag_count = Counter()
    # transition_tag = Counter()
    for sentence in train:
        a = True
        for wordtag in sentence:
            word = wordtag[0]
            tag = wordtag[1]
            tag_count[tag] += 1
            if a == True:
                if word not in initial_Tag:
                    initial_Tag[word] = Counter()
                    initial_Tag[word][tag] += 1
                else:
                    initial_Tag[word][tag] += 1
                # initial_Tag[tag] += 1
                a = False
            if word in counttagofeachword:
                counttagofeachword[word][tag] += 1
            else:
                counttagofeachword[word] = Counter()
                counttagofeachword[word][tag] += 1

    transition_tag = Counter()
    for sentence in train:
        for i in range(len(sentence) - 1):
            word1 = sentence[i][0]
            tag1 = sentence[i][1]
            word2 = sentence[i+1][0]
            tag2 = sentence[i+1][1]
            if tag1 not in transition_tag:
                transition_tag[tag1] = Counter()
                transition_tag[tag1][tag2] += 1
            else:
                transition_tag[tag1][tag2] += 1

    #initial prob
    # initial_Tag_prob = Counter()
    initial_smooth = 0.01
    for word in initial_Tag:
        for tag in initial_Tag[word]:
            # print(initial_Tag[word])
            initial_Tag[word][tag] = np.log((initial_Tag[word][tag] + initial_smooth)/(len(train) + initial_smooth*2))
            # print(initial_Tag[word][tag])
    unseen_initial_prob = np.log(initial_smooth/(len(train) + initial_smooth*2))

    tagcount = []
    for key in tag_count:
        if key != 'START':
            tagcount.append(key)
    # print(len(tagcount))

    transition_smooth = 0.0001
    #transition prob
    # transition_prob = np.zeros(shape=(len(tag_count), len(tag_count)))
    for tag1 in transition_tag:
        for tag2 in transition_tag[tag1]:
            transition_tag[tag1][tag2] = np.log((transition_tag[tag1][tag2] + transition_smooth)/(tag_count[tag1]
            + transition_smooth * (len(tagcount) + 1)))

    hapax = Counter()
    # print(counttagofeachword)
    for word in counttagofeachword:
        if len(counttagofeachword[word]) == 1:
            for tag in counttagofeachword[word]:
                if counttagofeachword[word][tag] == 1:
                    hapax[tag] += 1

    emission_smooth = 0.0001
    for word in counttagofeachword:
        for tag in counttagofeachword[word]:
            counttagofeachword[word][tag] = np.log((counttagofeachword[word][tag] + emission_smooth)/(tag_count[tag]
            + emission_smooth * (len(tag_count)+1)))

    for sentence in test:
        prob = []
        firstword = (0,0)
        prev = Counter()
        for i in range(len(sentence)):
            prob_tag = []
            word = sentence[i]
            if i == 0:
                if word in initial_Tag:
                    for tag in initial_Tag[word]:
                        proba = initial_Tag[word][tag] + counttagofeachword[word][tag]
                        prob_tag.append((proba, 'START'))
                        firstword = (proba, 'START')
                else:
                    proba = unseen_initial_prob + np.log((emission_smooth)/(tag_count[tag]
                    + emission_smooth * (len(tag_count)+1)))
                    prob_tag.append((proba, 'START'))
                    firstword = (proba, 'START')
            else:
                probe = 0.1
                probt = 0.1
                for tag in tagcount:
                    if word in counttagofeachword:
                        if tag in counttagofeachword[word]:
                            probe = counttagofeachword[word][tag]
                        else:
                            probe =  np.log((emission_smooth)/(tag_count[tag]
                            + emission_smooth * (len(tag_count)+1)))
                    else:
                        if tag in hapax:
                            probe = np.log(emission_smooth*(hapax[tag]/(sum(hapax.values()))))
                        else:
                            probe = np.log(emission_smooth/(sum(hapax.values())))
                    maxv = -999999
                    maxpre = (-9999999,'START')
                    for tuple in prob[i-1]:
                        p = tuple[0]
                        x = tuple[1]
                        if x in transition_tag:
                            if tag in transition_tag[x]:
                                probt = transition_tag[x][tag]
                            else:
                                probt = np.log((transition_smooth)/(tag_count[x]
                                + transition_smooth * (len(tagcount) + 1)))
                        else:
                            probt = np.log((transition_smooth)/(tag_count[x]
                            + transition_smooth * (len(tagcount)+1)))
                        proba = p + probe + probt
                        if proba > maxv:
                            maxv = proba
                            maxpre = tuple
                    prev[(maxv, tag)] = maxpre
                    prob_tag.append((maxv, tag))
                    # print(prob_tag)

            prob.append(prob_tag)
            if (len(prob) == len(sentence)):
                max_tuple = (-99999999, 0)
                for tuple in prob[len(prob) - 1]:
                    if tuple[0] >= max_tuple[0]:
                        max_tuple = tuple
                h = max_tuple
                predic = []
                predic.append(h[1])
                while (h in prev and prev[h][1] != firstword[1]):
                    h = prev[h]
                    # if (h[1] == 'START'):
                    #     break
                    predic.append(h[1])
                predic.append(firstword[1])
                predic.reverse()
                predict = []
                for i in range(len(sentence)):
                    predict.append((sentence[i], predic[i]))
                predicts.append(predict)
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)
    return predicts

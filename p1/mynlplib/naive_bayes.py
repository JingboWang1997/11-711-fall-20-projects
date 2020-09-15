from mynlplib.constants import OFFSET
from mynlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    combined = defaultdict(int)
    for i, yi in enumerate(y):
        if yi != label:
            continue
        for e in x[i]:
            combined[e] += x[i][e]
    return combined

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    res = {}
    corpus_counts = get_corpus_counts(x, y, label)
    total = sum(corpus_counts.values()) + (len(vocab) * smoothing)
    for v in vocab:
        res[v] = np.log((corpus_counts[v] + smoothing) / total)
    return defaultdict(float, res)

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    
    vocab = []
    for d in x:
        vocab += d.keys()
    vocab = list(set(vocab))
    labels = set(y)
   
    weights = defaultdict(float)
    for label in labels:
        pxy = estimate_pxy(x,y,label,smoothing,vocab)
        for k in pxy.keys():
            weights[(label, k)] = pxy[k]
        weights[(label, OFFSET)] = np.log(list(y).count(label) / len(y))
        
    
    return weights

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    '''

    labels = set(y_dv)
    scores = {}
    for s in smoothers:
        weights = estimate_nb(x_tr, y_tr, s)
        y_hat = clf_base.predict_all(x_dv, weights, labels)
        scores[s] = evaluation.acc(y_hat,y_dv)
    return clf_base.argmax(scores), scores

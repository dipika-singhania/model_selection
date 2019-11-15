""" Set of utilities """
import numpy as np


class ValueMeter(object):
    def __init__(self):
        self.sum = 0
        self.total = 0

    def add(self, value, n):
        self.sum += value*n
        self.total += n

    def value(self):
        return self.sum / self.total


class ArrayValueMeter(object):
    def __init__(self, dim=1):
        self.sum = np.zeros(dim)
        self.total = 0

    def add(self, arr, n):
        self.sum += arr*n
        self.total += n

    def value(self):
        val = self.sum / self.total
        if len(val) == 1:
            return val[0]
        else:
            return val


def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        rankings: numpy ndarray, shape = (instance_count, label_count)
        labels: numpy ndarray, shape = (instance_count,)
        ks: tuple of integers

    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
    rankings = scores.argsort()[:, ::-1]
    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]

def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels)
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0
    #np.zeros((scores.shape[0], scores.shape[1]))
    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
    return recalls/len(classes)



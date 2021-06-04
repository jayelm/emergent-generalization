"""
Tools for measuring emergence of communication
"""


from typing import Callable, Union
from scipy.spatial import distance
from scipy.stats import spearmanr
import editdistance

from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import torch


def python_pdist(X, metric, **kwargs):
    """
    From https://github.com/scipy/scipy/blob/v1.6.0/scipy/spatial/distance.py#L2057-L2069
    """
    m = len(X)
    k = 0
    dm = np.empty((m * (m - 1)) // 2, dtype=np.double)
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            dm[k] = metric(X[i], X[j], **kwargs)
            k = k + 1
    return dm


def edit_distance(x, y):
    return editdistance.eval(x, y) / ((len(x) + len(y)) / 2)


def topsim(
    meanings: torch.Tensor,
    messages: torch.Tensor,
    meaning_distance_fn: Union[str, Callable] = "hamming",
    message_distance_fn: Union[str, Callable] = "edit",
) -> float:
    """
    This function taken from EGG
    https://github.com/facebookresearch/EGG/blob/ace483e30c99a5bc480d84141bcc6f4416e5ec2b/egg/core/language_analysis.py#L164-L199
    but modified to allow pure python pdist with lists when a distance fn is
    callable (rather than scipy coercing to 2d arrays)
    """

    distances = {
        "cosine": distance.cosine,
        "hamming": distance.hamming,
        "jaccard": distance.jaccard,
        "euclidean": distance.euclidean,
    }

    slow_meaning_fn = True
    if meaning_distance_fn in distances:
        meaning_distance_fn_callable = distances[meaning_distance_fn]
        slow_meaning_fn = False
    elif meaning_distance_fn == "edit":
        meaning_distance_fn_callable = edit_distance
    else:
        meaning_distance_fn_callable = meaning_distance_fn

    slow_message_fn = True
    if message_distance_fn in distances:
        message_distance_fn_callable = distances[message_distance_fn]
        slow_message_fn = False
    elif message_distance_fn == "edit":
        message_distance_fn_callable = edit_distance
    else:
        message_distance_fn_callable = message_distance_fn

    assert (
        meaning_distance_fn_callable and message_distance_fn_callable
    ), f"Cannot recognize {meaning_distance_fn} \
        or {message_distance_fn} distances"

    # If meaning distance fn is not a scipy func
    if slow_meaning_fn:
        meaning_dist = python_pdist(meanings, meaning_distance_fn_callable)
    else:
        meaning_dist = distance.pdist(meanings, meaning_distance_fn_callable)

    if slow_message_fn:
        message_dist = python_pdist(messages, message_distance_fn_callable)
    else:
        message_dist = distance.pdist(messages, message_distance_fn_callable)

    topsim = spearmanr(meaning_dist, message_dist, nan_policy="raise").correlation

    return topsim


def normalize(ctr):
    total = sum(ctr.values())
    return Counter({k: v / total for k, v in ctr.items()})


def context_independence(concepts, messages):
    r"""
    Measure context independence between concepts c and messages m.

    Let p_cm(c | m) be the conditional probability of context c given message m and
    p_mc(m | c) be the condiational probability of message m given context c
    (we can estimate these by simply enumerating).

    Then for any concept c, we define the "ideal" message m^c as argmax_m
    p_cm(c | m) (i.e., whichever message has the highest conditional
    probability that we are referring to concept c).

    Then,

    CI(concepts, messages) = 1 / len(concepts) \sum_c p_mc(m^c | c) * p_cm(c | m^c).
    """
    p_cm = defaultdict(Counter)
    p_mc = defaultdict(Counter)

    for c, m in zip(concepts, messages):
        # I.e., given message m, conditional probability of c.
        p_cm[m][c] += 1
        p_mc[c][m] += 1

    p_cm = {k: normalize(v) for k, v in p_cm.items()}
    p_mc = {k: normalize(v) for k, v in p_mc.items()}

    # Find ideal messages
    unique_concepts = list(set(concepts))
    unique_messages = list(set(messages))
    cis = []

    for c in unique_concepts:
        mc = None
        best_p_cm = 0.0
        for m in unique_messages:
            this_p_cm = p_cm[m][c]
            if this_p_cm > best_p_cm:
                mc = m
                best_p_cm = this_p_cm
        if mc is None:
            raise RuntimeError(f"Couldn't find ideal concept for {c}")

        ci = p_mc[c][mc] * p_cm[mc][c]

        cis.append(ci)

    return np.mean(cis)


def mutual_information(concepts, messages):
    r"""
    Measure mutual information between concepts c and messages m (assuming
    enumerability)
    """
    # Assign int values
    c2i = {}
    m2i = {}

    for c, m in zip(concepts, messages):
        if c not in c2i:
            c2i[c] = len(c2i)
        if m not in m2i:
            m2i[m] = len(m2i)

    cis = [c2i[c] for c in concepts]
    mis = [m2i[m] for m in messages]

    return normalized_mutual_info_score(cis, mis)

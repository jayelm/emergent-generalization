"""
utils
"""


import os
import subprocess
import json
import warnings
import numpy as np
import io_util
from PIL import Image
import torch
from collections import defaultdict


def current_git_hash():
    """
    Get the hash of the latest commit in this repository. Does not account for unstaged changes.
    Returns
    -------
    git_hash : ``str``, optional
        The string corresponding to the current git hash if known, else ``None`` if something failed.
    """
    unstaged_changes = False
    try:
        subprocess.check_output(["git", "diff-index", "--quiet", "HEAD", "--"])
    except subprocess.CalledProcessError as grepexc:
        if grepexc.returncode == 1:
            warnings.warn("Running experiments with unstaged changes.")
            unstaged_changes = True
    except FileNotFoundError:
        warnings.warn("Git not found")
    try:
        git_hash = (
            subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode("utf-8")
        )
        return git_hash, unstaged_changes
    except subprocess.CalledProcessError:
        return None, None


class Statistics:
    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, batch_size=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, batch_size)

    def averages(self):
        """
        Compute averages from meters. Handle tensors vs floats (always return a
        float)

        Parameters
        ----------
        meters : Dict[str, util.AverageMeter]
            Dict of average meters, whose averages may be of type ``float`` or ``torch.Tensor``

        Returns
        -------
        metrics : Dict[str, float]
            Average value of each metric
        """
        metrics = {m: vs.avg for m, vs in self.meters.items()}
        metrics = {
            m: v if isinstance(v, float) else v.item() for m, v in metrics.items()
        }
        return metrics

    def __str__(self):
        meter_str = ", ".join(f"{k}={v}" for k, v in self.meters.items())
        return f"Statistics({meter_str})"


class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self, running_avg=False):
        self.reset()
        self.compute_running_avg = running_avg
        if self.compute_running_avg:
            self.reset_running_avg()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset_running_avg(self):
        self.running_val = 0
        self.running_avg = 0
        self.running_sum = 0
        self.running_count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.compute_running_avg:
            self.update_running_avg(val, n)

    def update_running_avg(self, val, n):
        self.running_val = val
        self.running_sum += val * n
        self.running_count += n
        self.running_avg = self.running_sum / self.running_count

    def __str__(self):
        return f"AverageMeter(mean={self.avg:f}, count={self.count:d})"

    def __repr__(self):
        return str(self)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    scores and labels are 2d predictions, and we don't care about seqs of
    predictions. So if this is a seq prediction task, we give partial credit

    :param scores: torch.Tensor scores from the model
    :param targets: torch.Tensor true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def compute_average_metrics(meters):
    """
    Compute averages from meters. Handle tensors vs floats (always return a
    float)

    Parameters
    ----------
    meters : Dict[str, util.AverageMeter]
        Dict of average meters, whose averages may be of type ``float`` or
        ``torch.Tensor``

    Returns
    -------
    metrics : Dict[str, float]
        Average value of each metric
    """
    metrics = {m: vs.avg for m, vs in meters.items()}
    metrics = {
        m: float(v) if isinstance(v, float) or isinstance(v, int) else v.item()
        for m, v in metrics.items()
    }
    return metrics


def save_args(args, exp_dir, filename="args.json"):
    args_dict = vars(args)
    args_dict["git_hash"], args_dict["git_unstaged_changes"] = current_git_hash()
    with open(os.path.join(exp_dir, filename), "w") as f:
        json.dump(args_dict, f, indent=4, separators=(",", ": "), sort_keys=True)


def load_args(exp_dir, filename="args.json"):
    with open(os.path.join(exp_dir, filename), "r") as f:
        args = json.load(f)
        # Delete git hash
        if "git_hash" in args:
            del args["git_hash"]
        if "git_unstaged_changes" in args:
            del args["git_unstaged_changes"]
    return args


def restore_missing_defaults(args, verbose=False):
    defaults = io_util.parse_args(defaults=True)
    defaults = vars(defaults)
    for missing_attr, default_value in defaults.items():
        if not hasattr(args, missing_attr):
            if verbose:
                print("Restoring {missing_attr}={default_value}")
            setattr(args, missing_attr, default_value)


# Debug utilities
def dsave(x, f):
    Image.fromarray(np.transpose(x, (1, 2, 0))).save(f)


def dtext(x, dataset):
    return dataset.to_text([x])


def update_with_prefix(d, new_d, prefix):
    d.update({f"{prefix}_{k}": v for k, v in new_d.items()})


def to_emergent_text(idxs, join=False, eos=None):
    texts = []
    for lang in idxs:
        toks = []
        for i in lang:
            i_item = i.item()
            i = str(i_item)
            toks.append(i)
            if eos is not None and i_item == eos:
                break
        if join:
            texts.append(" ".join(toks))
        else:
            texts.append(toks)
    return texts


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import hausdorff
import numba
import itertools


def stack_pos_neg(pos_imgs, neg_imgs):
    """
    Given positive images and negative images, stack them, then create a y tensor of labels.

    TODO - instead of stack just assign?
    """
    pos_imgs = torch.stack(pos_imgs)
    neg_imgs = torch.stack(neg_imgs)

    imgs = torch.cat([pos_imgs, neg_imgs], 0)
    y = torch.zeros(imgs.shape[0], dtype=torch.uint8)
    y[: pos_imgs.shape[0]] = 1
    return imgs, y


def split_spk_lis(inp, y, n_examples, percent_novel=1.0):
    midp = inp.shape[0] // 2
    n_pos_ex = n_examples // 2

    spk_inp = torch.zeros((n_examples, *inp.shape[1:]), dtype=inp.dtype)
    spk_inp[:n_pos_ex] = inp[:n_pos_ex]
    spk_inp[n_pos_ex:] = inp[midp : midp + n_pos_ex]

    spk_label = torch.zeros(n_examples, dtype=torch.uint8)
    spk_label[:n_pos_ex] = 1

    lis_inp = torch.zeros((n_examples, *inp.shape[1:]), dtype=inp.dtype)
    lis_inp[:n_pos_ex] = inp[n_pos_ex : 2 * n_pos_ex]
    lis_inp[n_pos_ex:] = inp[midp + n_pos_ex : midp + (2 * n_pos_ex)]

    lis_label = torch.zeros(n_examples, dtype=torch.uint8)
    lis_label[:n_pos_ex] = 1

    if percent_novel == 0.0:
        lis_inp = spk_inp
        lis_label = spk_label
    elif percent_novel < 1.0:  # Sample some negatives
        is_novel = torch.rand(n_pos_ex) < percent_novel
        if spk_inp.ndim == 4:  # Image
            is_novel_exp = is_novel.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:  # Feat
            is_novel_exp = is_novel.unsqueeze(1)

        lis_inp = torch.where(is_novel_exp, lis_inp, spk_inp)
        lis_label = torch.where(is_novel, lis_label, spk_label)

    return spk_inp, spk_label, lis_inp, lis_label


def train_test_split_pt(*tensors, test_size=0.2):
    """
    Pytorch train test split
    """
    first = tensors[0]
    if isinstance(test_size, float):
        test_size = int(first.shape[1] * test_size)
    perm = torch.randperm(first.shape[1])

    train_perm = []
    test_perm = []
    for t in tensors:
        t_perm = t[:, perm]
        train_perm.append(t_perm[:, test_size:].contiguous())
        test_perm.append(t_perm[:, :test_size].contiguous())

    return train_perm, test_perm


def train_val_test_split(data, val_size=0.1, test_size=0.1, random_state=None):
    """
    Split data into train, validation, and test splits
    Parameters
    ----------
    data : ``np.Array``
        Data of shape (n_data, 2), first column is ``x``, second column is ``y``
    val_size : ``float``, optional (default: 0.1)
        % to reserve for validation
    test_size : ``float``, optional (default: 0.1)
        % to reserve for test
    random_state : ``np.random.RandomState``, optional (default: None)
        If specified, random state for reproducibility
    """
    idx = np.arange(data["imgs"].shape[0])
    idx_train, idx_valtest = train_test_split(
        idx, test_size=val_size + test_size, random_state=random_state, shuffle=True
    )
    idx_val, idx_test = train_test_split(
        idx_valtest,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
        shuffle=True,
    )
    splits = []
    for idx_split in (idx_train, idx_val, idx_test):
        splits.append(
            {
                "imgs": data["imgs"][idx_split],
                "labels": data["labels"][idx_split],
                "langs": data["langs"][idx_split],
            }
        )
    return splits


def return_index(getitem):
    def with_index(self, index):
        res = getitem(self, index)
        return res + (index,)

    return with_index


@numba.jit(nopython=True, fastmath=True)
def hamming(x, y):
    """
    From https://github.com/talboger/fastdist/blob/master/fastdist/fastdist.py
    """
    n = len(x)
    num, denom = 0, 0
    for i in range(n):
        if x[i] != y[i]:
            num += 1
        denom += 1
    return num / denom


def get_pairwise_hausdorff_distances(concepts):
    dists = {}
    pairs = itertools.combinations_with_replacement(sorted(concepts.items()), 2)
    for (c1, a1), (c2, a2) in pairs:
        if (c1, c2) not in dists:
            dists[(c1, c2)] = hausdorff.hausdorff_distance(a1, a2, distance=hamming)
    return dists


def get_game_type(args):
    if args.reference_game:
        return "ref"
    elif args.percent_novel == 0.0:
        return "setref"
    elif args.percent_novel == 1.0:
        return "concept"
    else:
        return None

import numpy as np
from torch.utils.data import DataLoader
import torch

from . import shapeworld
from . import cub


def load(args, **kwargs):
    if "shapeworld" in args.dataset:
        lf = shapeworld.load
    elif "cub" in args.dataset:
        lf = cub.load
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return lf(args, **kwargs)


def worker_init(worker_id):
    np.random.seed()
    torch.seed()


def load_dataloaders(args, **kwargs):
    datas = load(args, **kwargs)

    def to_dl(dset):
        return DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=True,
            worker_init_fn=worker_init,
        )

    dataloaders = {split: to_dl(dset) for split, dset in datas.items()}

    if args.test_dataset is not None:
        # TODO: context manage this
        orig_dataset = args.dataset
        orig_percent_novel = args.percent_novel
        orig_n_examples = args.n_examples

        args.dataset = args.test_dataset
        args.percent_novel = args.test_percent_novel
        args.n_examples = args.test_n_examples

        test_datas = load(args)

        args.dataset = orig_dataset
        args.percent_novel = orig_percent_novel
        args.n_examples = orig_n_examples

        dataloaders["test"] = to_dl(test_datas["test"])

    return dataloaders

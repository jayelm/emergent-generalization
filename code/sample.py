"""
Sample from a pretrained speaker, correcting for train/test imbalance (check #
of concepts)
"""

import os

import torch
import pandas as pd

import util
import models
import data
from train import run
import json
from tqdm import tqdm


def sample(pair, dataloaders, exp_args, args):
    dname = dataloaders["train"].dataset.name
    if dname == "cub":
        n_train_concepts = 100
        n_test_concepts = 50
    elif dname == "shapeworld":
        # Note - ref game has different numbers, but the same split
        n_train_concepts = 312
        n_test_concepts = 63
    else:
        raise NotImplementedError(dname)

    train_pct = n_train_concepts / (n_train_concepts + n_test_concepts)
    n_train_samples = round(train_pct * args.n)
    n_test_samples = args.n - n_train_samples

    def sample_split(split, n):
        stats = util.Statistics()
        all_lang = pd.DataFrame()
        pbar = tqdm(desc=f"sample {split}", total=n)
        while all_lang.shape[0] < n:
            split_stats, lang = run(
                split, 0, pair, None, dataloaders, exp_args, force_no_train=True
            )
            if dname == "cub":  # Zero out metadata
                lang["md"] = 0
            all_lang = pd.concat((all_lang, lang))
            pbar.update(lang.shape[0])
            stats.update(**split_stats)
        pbar.close()
        all_lang = all_lang.head(n)
        all_lang["split"] = split
        stats = stats.averages()
        return stats, all_lang

    train_stats, train_lang = sample_split("train", n_train_samples)
    test_stats, test_lang = sample_split("test", n_test_samples)

    if args.force_reference_game:
        lang_fname = "sampled_lang_force_ref.csv"
    elif args.force_concept_game:
        lang_fname = "sampled_lang_force_concept.csv"
    elif args.force_setref_game:
        lang_fname = "sampled_lang_force_setref.csv"
    else:
        lang_fname = "sampled_lang.csv"
    lang_fname = os.path.join(args.exp_dir, lang_fname)
    all_lang = pd.concat((train_lang, test_lang))
    all_lang.to_csv(lang_fname, index=False)

    # Save statistics
    comb_stats = {}
    util.update_with_prefix(comb_stats, train_stats, "train")
    util.update_with_prefix(comb_stats, test_stats, "test")
    if args.force_reference_game:
        fname = "sampled_stats_force_ref.json"
    elif args.force_concept_game:
        fname = "sampled_stats_force_concept.json"
    elif args.force_setref_game:
        fname = "sampled_stats_force_setref.json"
    else:
        fname = "sampled_stats.json"
    with open(os.path.join(args.exp_dir, fname), "w") as f:
        json.dump(comb_stats, f)


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("exp_dir")
    parser.add_argument("--n", default=200000, type=int, help="Number of samples")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force_reference_game", action="store_true")
    group.add_argument("--force_concept_game", action="store_true")
    group.add_argument("--force_setref_game", action="store_true")

    args = parser.parse_args()

    # Restore args + defaults if missing
    exp_args = Namespace(**util.load_args(args.exp_dir))
    util.restore_missing_defaults(exp_args)

    if args.force_reference_game:
        exp_args.reference_game = True
        exp_args.percent_novel = 0.0
        if (
            "shapeworld" in exp_args.dataset
            and "shapeworld_ref" not in exp_args.dataset
        ):
            # Change SW dataset to ref version (no change for CUB)
            if "shapeworld_all" in exp_args.dataset:
                exp_args.dataset = exp_args.dataset.replace(
                    "shapeworld_all", "shapeworld_ref"
                )
            else:
                exp_args.dataset = exp_args.dataset.replace(
                    "shapeworld", "shapeworld_ref"
                )
    elif args.force_concept_game:
        exp_args.reference_game = False
        exp_args.percent_novel = 1.0
        if "shapeworld_ref" in exp_args.dataset:
            # Change SW dataset to ref version (no change for CUB)
            exp_args.dataset = exp_args.dataset.replace("shapeworld_ref", "shapeworld")
    elif args.force_setref_game:
        exp_args.reference_game = False
        exp_args.percent_novel = 0.0
        if "shapeworld_ref" in exp_args.dataset:
            # Change SW dataset to ref version (no change for CUB)
            exp_args.dataset = exp_args.dataset.replace("shapeworld_ref", "shapeworld")

    dataloaders = data.loader.load_dataloaders(exp_args)
    model_config = models.builder.build_models(dataloaders, exp_args)
    pair = model_config["pair"]

    state_dict = torch.load(os.path.join(args.exp_dir, "best_model.pt"))
    pair.load_state_dict(state_dict)

    sample(pair, dataloaders, exp_args, args)

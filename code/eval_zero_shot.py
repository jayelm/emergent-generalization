"""
Zero shot eval according to composed utterances
"""

import os

import vis
import torch
import numpy as np
import json
from collections import defaultdict, Counter
from torch.nn.utils.rnn import pad_sequence
import random
import torch.nn.functional as F
import pandas as pd
from scipy.spatial.distance import squareform
from tqdm import tqdm
import scipy.stats

import util
import models
import data
from data.shapeworld import concept_to_lf
from tqdm import trange
import emergence


from nltk.translate.bleu_score import sentence_bleu


def sentence_bleu1(references, hypothesis):
    return sentence_bleu(references, hypothesis, weights=(1.0,))


def get_acre_lang(sampled_lang):
    acre_lang = defaultdict(lambda: defaultdict(list))
    for i, concept in enumerate(sampled_lang["gt_lang"]):
        acre_lang[concept]["acre"].append(
            (
                sampled_lang["pred_lang"][i],
                sampled_lang["pred_lang_len"][i],
            )
        )

        acre_lang[concept]["ground_truth_1"].append(
            (
                sampled_lang["model_lang"][i],
                sampled_lang["model_lang_len"][i],
            )
        )
    return acre_lang


def sample_acre_lang(true_lang, acre_lang, dataset, vocab_size):
    true_lang = dataset.to_text(true_lang)
    langs = {}
    ltypes = ["acre", "ground_truth_1"]
    # NOTE - we don't have language for the unseen primitives...what to do?
    # Reform the batch I think? (Where None)
    for ltype in ltypes:
        sampled_lang = []
        sampled_lang_len = []
        for c in true_lang:
            # Remove <s>, </s>
            c = c[4:-5]
            if c in acre_lang:
                s_lang, s_lang_len = random.choice(acre_lang[c][ltype])
            else:
                s_lang = torch.zeros(1, dtype=torch.int64)
                s_lang_len = 0

            sampled_lang.append(s_lang)
            sampled_lang_len.append(s_lang_len)

        lang_pad = pad_sequence(sampled_lang, batch_first=True)
        lang_len = torch.tensor(sampled_lang_len)
        # Trim language length
        lang_pad = lang_pad[:, : lang_len.max()]
        # To onehot
        lang_pad = F.one_hot(lang_pad, num_classes=vocab_size)
        lang_pad = lang_pad.float()

        langs[ltype] = (lang_pad, lang_len)

    return langs


def randstrlen(maxlen, nchar=26):
    assert maxlen >= 1
    size = maxlen
    while size != 1:
        if random.randrange(nchar + 1) != 0:
            return size
        size -= 1
    return 1


def randstr(maxlen, nchar=26):
    strlen = randstrlen(maxlen, nchar=nchar)
    chars = [data.language.SOS_IDX]
    for i in range(strlen):
        new_char = 2
        while new_char == 2:
            new_char = random.randrange(nchar)
        chars.append(new_char)
    chars.append(data.language.EOS_IDX)
    chars = np.array(chars)
    return chars, strlen + 2


def emergent_to_idx(langs):
    # Add SOS, EOS
    lang_len = np.array([len(t) for t in langs], dtype=np.int)
    lang_idx = np.full((len(langs), max(lang_len)), data.language.PAD_IDX, dtype=np.int)
    for i, toks in enumerate(langs):
        for j, tok in enumerate(toks):
            lang_idx[i, j] = int(tok)
    return lang_idx, lang_len


def pairs_to_lang(pairs, vocab_size):
    lang, lang_len = zip(*pairs)
    lang_len = torch.tensor(lang_len)
    lang = [torch.tensor(x) for x in lang]
    lang = pad_sequence(lang, batch_first=True)
    lang = F.one_hot(lang, num_classes=vocab_size)
    lang = lang.float()
    return lang, lang_len


def get_lang_per_concept(exp_dir):
    slang = pd.read_csv(
        os.path.join(exp_dir, "sampled_lang.csv"), keep_default_na=False
    )
    emergent_idx = emergent_to_idx(slang["lang"].str.strip().str.split(" "))
    # From concepts to language
    lang_per_concept = defaultdict(list)
    for i, concept in enumerate(slang["true_lang"]):
        lang_per_concept[concept].append((emergent_idx[0][i], emergent_idx[1][i]))
    return dict(lang_per_concept)


def sample_other_lang_from_concept(concepts, lang_per_concept, vocab_size):
    concept_langs = []
    for c in concepts:
        c = c[4:-5]
        concept_langs.append(random.choice(lang_per_concept[c]))
    return pairs_to_lang(concept_langs, vocab_size)


def sample_other_lang(n, lang_per_concept, vocab_size):
    # List of values uniformly
    all_lang = []
    for concept_langs in lang_per_concept.values():
        all_lang.extend(concept_langs)

    langs = []
    for _ in range(n):
        langs.append(random.choice(all_lang))
    return pairs_to_lang(langs, vocab_size)


def sample_other_lang_from_closest_concept(
    concepts, concept_distances, lang_per_concept, vocab_size
):
    closests = []
    for c in concepts:
        c = c[4:-5]
        c_nn = concept_distances.get_closest_concept(c)
        closests.append(random.choice(lang_per_concept[c_nn]))
    return pairs_to_lang(closests, vocab_size)


def sample_rand_unif_lang(n, max_lang_length, vocab_size):
    rands = []
    for _ in range(n):
        # Do minus 2, since we need to include sos/eos
        randpair = randstr(max_lang_length - 2, nchar=vocab_size)
        rands.append(randpair)
    return pairs_to_lang(rands, vocab_size)


class PairwiseDistances:
    def __init__(self, concepts):
        # Split
        self.c2i = {v: k for k, v in enumerate(concepts)}
        self.i2c = dict(enumerate(concepts))
        self.concept_tuples = [tuple(c.split(" ")) for c in concepts]
        self.distances = emergence.python_pdist(
            self.concept_tuples, emergence.edit_distance
        )
        self.distances = squareform(self.distances)

    def __getitem__(self, pair):
        c1, c2 = pair
        i1 = self.c2i[c1]
        i2 = self.c2i[c2]
        return self.distances[i1, i2]

    def get_closest_concept(self, c):
        """
        Get closest concept to this one by hamming distance, breaking ties randomly
        """
        i = self.c2i[c]
        # TEMP HACK - set distance with self to be inf
        dists = self.distances[i]

        # Choose the right distance
        assert dists[i] == 0.0
        dists[i] = np.inf
        c_nn = dists.argmin()
        dist = dists[c_nn]
        all_nn = np.argwhere(np.isclose(dists, dist)).squeeze(1)

        # Reassign original distance
        dists[i] = 0.0

        samp_nn = random.choice(all_nn)
        return self.i2c[samp_nn]


def get_acre_split(exp_dir):
    with open(os.path.join(exp_dir, "acre_split.json"), "r") as f:
        acre_split = json.load(f)
    # Map to tuples
    return {
        "train": set(to_tuple(acre_split["train"])),
        "test": set(to_tuple(acre_split["test"])),
    }


def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def get_lang_per_concept_counts(lpc):
    lpc_counts = defaultdict(Counter)
    for concept, langs in lpc.items():
        for lang, lang_len in langs:
            lang_str = " ".join(map(str, lang))
            lpc_counts[concept][lang_str] += 1
    return lpc_counts


def compute_metrics_by_concept(concepts, **kwargs):
    metrics_by_concept = defaultdict(lambda: defaultdict(list))
    for i, c in enumerate(concepts):
        for metric, vals in kwargs.items():
            metrics_by_concept[c][metric].append(vals[i])
    return metrics_by_concept


def eval_zero_shot(
    split, pair, dataloaders, sampled_lang, exp_args, args, epochs=1, n_vis=500
):
    other_split = "test" if split == "train" else "train"
    pair.eval()

    acre_lang = get_acre_lang(sampled_lang)

    stats = util.Statistics()

    lang_per_concept = get_lang_per_concept(args.exp_dir)
    lpc_counts = get_lang_per_concept_counts(lang_per_concept)
    lpc_unique = {
        c: list(set(list(lang_counts.keys()))) for c, lang_counts in lpc_counts.items()
    }
    # lang_type -> {concept -> counts}
    lang_types_per_concept = defaultdict(lambda: defaultdict(Counter))
    # Pairwise distances between language
    all_concepts = list(lang_per_concept.keys())
    concept_distances = PairwiseDistances(all_concepts)

    vocab_size = exp_args.vocab_size + 3
    acre_split = get_acre_split(args.exp_dir)

    for epoch in trange(epochs, desc="Eval"):
        n_vis_so_far = 0
        vis_spk_inp = []
        vis_spk_y = []
        vis_lis_inp = []
        vis_lis_y = []
        vis_true_lang = []
        vis_lang = defaultdict(list)
        vis_lis_pred = defaultdict(list)

        # This is separate from acre split
        for dataloader_split in ["train", "val", "test"]:
            dataloader = dataloaders[dataloader_split]
            for batch_i, batch in enumerate(tqdm(dataloader, desc=dataloader_split)):
                spk_inp, spk_y, lis_inp, lis_y, true_lang, md, idx = batch

                if dataloader.dataset.name == "shapeworld":
                    spk_inp = spk_inp.float() / 255
                    lis_inp = lis_inp.float() / 255
                else:
                    spk_inp = spk_inp.float()
                    lis_inp = lis_inp.float()
                spk_y = spk_y.float()
                lis_y = lis_y.float()
                if args.cuda:
                    spk_inp = spk_inp.cuda()
                    lis_inp = lis_inp.cuda()
                    lis_y = lis_y.cuda()
                    spk_y = spk_y.cuda()

                # Get acre predicted lang configurations
                langs = sample_acre_lang(
                    true_lang,
                    acre_lang,
                    dataloader.dataset,
                    vocab_size,
                )
                # Keep only language that we have model predictions for (for
                # this split)
                has_language = langs["acre"][1] != 0
                if not has_language.any():
                    continue

                spk_inp = spk_inp[has_language]
                spk_y = spk_y[has_language]
                lis_inp = lis_inp[has_language]
                lis_y = lis_y[has_language]
                true_lang = true_lang[has_language]

                # Batch size after language has been filtered out
                batch_size = spk_inp.shape[0]
                # Concepts
                concepts = dataloader.dataset.to_text(true_lang)

                # Assert all concepts for which we have language are in this split
                for cncpt in concepts:
                    cncpt = cncpt[4:-5]
                    c_lf = concept_to_lf(cncpt)
                    assert c_lf in acre_split[split], cncpt
                    assert c_lf not in acre_split[other_split], cncpt

                # Language sampled from other model utterances from the same concept
                langs["same_concept"] = sample_other_lang_from_concept(
                    concepts,
                    lang_per_concept,
                    vocab_size,
                )

                # Random language uniformly sampled from strings
                langs["random"] = sample_rand_unif_lang(
                    batch_size,
                    exp_args.max_lang_length,
                    vocab_size,
                )

                # Language sampled from the entire agent vocab
                langs["any_concept"] = sample_other_lang(
                    batch_size,
                    lang_per_concept,
                    vocab_size,
                )

                # Language sampled from concept that is closest in edit distance
                langs["closest_concept"] = sample_other_lang_from_closest_concept(
                    concepts,
                    concept_distances,
                    lang_per_concept,
                    vocab_size,
                )

                # Also sample from speaker (sanity check)
                with torch.no_grad():
                    spk_lang, _ = pair.speaker(
                        spk_inp, spk_y, max_len=exp_args.max_lang_length
                    )
                    langs["ground_truth_2"] = spk_lang

                # Finally, filter out the original acre language
                for ltype in ["acre", "ground_truth_1"]:
                    langs[ltype] = (
                        langs[ltype][0][has_language],
                        langs[ltype][1][has_language],
                    )

                # Eval all language types
                for lang_type, (lang, lang_length) in langs.items():
                    lang_text = util.to_emergent_text(lang.argmax(2), join=True, eos=2)

                    # Calculate BLEU
                    if lang_type != "ground_truth_1":
                        # Calculate BLEU with ground truth lang type: TODO - need to collect.
                        # A good data structure to have:
                        # For each lang type, map concepts -> language.
                        # you could do unique language or ocunts (I guess counts).
                        # Do counts, then you can use unique language for bleu1 score
                        # How many passes do you need through the data?
                        bleu1 = 0.0
                        for clang, concept in zip(lang_text, concepts):
                            concept = concept[4:-5]
                            crefs = lpc_unique[concept]
                            # Convert refs to lang
                            crefs_toks = [c.split(" ")[1:-1] for c in crefs]
                            clang_toks = clang.split(" ")[1:-1]
                            bleu1 += sentence_bleu1(crefs_toks, clang_toks)
                        bleu1 /= len(concepts)
                    else:
                        bleu1 = 1.0

                    # Collect lang types per concept
                    for clang, concept in zip(lang_text, concepts):
                        lang_types_per_concept[lang_type][concept[4:-5]][clang] += 1

                    if args.cuda:
                        lang = lang.cuda()
                        lang_length = lang_length.cuda()

                    with torch.no_grad():
                        lis_scores = pair.listener(lis_inp, lang, lang_length)
                        this_loss = pair.bce_criterion(lis_scores, lis_y)

                    lis_pred = (lis_scores > 0).float()
                    per_game_acc = (lis_pred == lis_y).float().mean(1).cpu().numpy()
                    this_acc = per_game_acc.mean()
                    stats.update(
                        **{
                            f"{lang_type}_loss": this_loss,
                            f"{lang_type}_acc": this_acc,
                            f"{lang_type}_bleu": bleu1,
                        },
                        batch_size=lis_scores.shape[0],
                    )

                    # Metrics by concept
                    mbc = compute_metrics_by_concept(concepts, acc=per_game_acc)
                    for concept, concept_metrics in mbc.items():
                        for metric, cms in concept_metrics.items():
                            n_cm = len(cms)
                            cm_mean = np.mean(cms)
                            concept_underscore = concept[4:-5].replace(" ", "_")
                            stats.update(
                                **{
                                    f"{lang_type}_{concept_underscore}_{metric}": cm_mean
                                },
                                batch_size=n_cm,
                            )

                    # Save processed text
                    if n_vis_so_far < n_vis:
                        vis_lang[lang_type].append(lang_text)
                        vis_lis_pred[lang_type].append(lis_pred)

                stats.update(
                    n=lis_scores.shape[0],
                    batch_size=1,
                )

                # For visualization
                if n_vis_so_far < n_vis:
                    vis_true_lang.append(
                        dataloader.dataset.to_text(true_lang, join=True)
                    )
                    vis_spk_inp.append(spk_inp.cpu())
                    vis_spk_y.append(spk_y.cpu())
                    vis_lis_inp.append(lis_inp.cpu())
                    vis_lis_y.append(lis_y.cpu())

                n_vis_so_far += spk_inp.shape[0]

        # Flatten input for visualization
        vis_spk_inp = torch.cat(vis_spk_inp, 0)
        vis_spk_y = torch.cat(vis_spk_y, 0)
        vis_lis_inp = torch.cat(vis_lis_inp, 0)
        vis_lis_y = torch.cat(vis_lis_y, 0)
        vis_lang = {
            lang_type: [item for sublist in lang_text for item in sublist]
            for lang_type, lang_text in vis_lang.items()
        }
        vis_true_lang = [item for sublist in vis_true_lang for item in sublist]
        vis_lis_pred = {
            lang_type: torch.cat(lis_pred, 0)
            for lang_type, lis_pred in vis_lis_pred.items()
        }

        vis.report(
            vis_spk_inp,
            vis_spk_y,
            vis_lis_inp,
            vis_lis_y,
            dataloader.dataset,
            epoch,
            split,
            vis_lang,
            vis_true_lang,
            vis_lis_pred,
            exp_dir=os.path.join(args.exp_dir, "zero_shot"),
        )

    seen_str = "seen" if split == "train" else "unseen"
    lang_type_records = get_lang_type_records(lang_types_per_concept, seen=seen_str)
    return stats.averages(), lang_type_records


def get_lang_type_records(lang_types, **kwargs):
    records = []
    for lang_type, lt_concepts in lang_types.items():
        for concept, counts in lt_concepts.items():
            counts_total = sum(counts.values())
            counts_norm = {k: v / counts_total for k, v in counts.items()}
            entropy = scipy.stats.entropy(list(counts.values()))

            for lang, count in counts.items():
                records.append(
                    {
                        "concept": concept,
                        "lang": lang,
                        "count": count,
                        # I never use percent, so no need
                        #  "percent": counts_norm[lang],
                        "entropy": entropy,
                        "lang_type": lang_type,
                        **kwargs,
                    }
                )
    return records


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("exp_dir")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()

    # Restore
    exp_args = Namespace(**util.load_args(args.exp_dir))
    util.restore_missing_defaults(exp_args)
    dataloaders = data.loader.load_dataloaders(exp_args, fast=True)
    model_config = models.builder.build_models(dataloaders, exp_args)
    pair = model_config["pair"]
    if args.cuda:
        pair.cuda()

    state_dict = torch.load(os.path.join(args.exp_dir, "best_model.pt"))
    pair.load_state_dict(state_dict)

    # split here is acre split, but you should be evalling across all datasets...
    # maybe concatenate train, val, test?
    lang_type_records = []
    for split in ["train", "test"]:
        sampled_lang = torch.load(
            os.path.join(args.exp_dir, f"sampled_lang_{split}_sampled_lang.pt")
        )
        metrics, records = eval_zero_shot(
            split, pair, dataloaders, sampled_lang, exp_args, args, epochs=args.epochs
        )
        with open(os.path.join(args.exp_dir, f"zero_shot_{split}.json"), "w") as f:
            json.dump(metrics, f)
        lang_type_records.extend(records)

    pd.DataFrame(lang_type_records).to_csv(
        os.path.join(args.exp_dir, "zero_shot_lang_type_stats.csv"), index=False
    )

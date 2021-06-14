"""
Seq2seq-style ACRe
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm, trange
import string
import itertools
import os
import json
import scipy.stats
import random
from argparse import Namespace
import copy

from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from data import language
from data.shapeworld import concept_to_lf, lf_to_concept, get_unique_concepts
from data.loader import load_dataloaders
from models import seq2seq
import util


class AddFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y


class MeanFusion(AddFusion):
    def forward(self, x, y):
        res = super().forward(x, y)
        return res / 2.0


class MultiplyFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x * y


class MLPFusion(nn.Module):
    def __init__(self, x_size, n_layers=1):
        super().__init__()

        self.x_size = x_size
        self.n_layers = n_layers

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(self.x_size * 2, self.x_size))
            else:
                layers.append(nn.ReLU())
                layers.append(self.x_size, self.x_size)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, y):
        xy = torch.cat([x, y], 1)
        return self.mlp(xy)


FUSION_MODULES = {
    "add": AddFusion,
    "average": MeanFusion,
    "multiply": MultiplyFusion,
    "mlp": MLPFusion,
}


def get_transformer_encoder_decoder(
    vocab_size,
    embedding_size,
    hidden_size,
    intermediate_size,
    num_hidden_layers=2,
    num_attention_heads=2,
    max_position_embeddings=60,
    hidden_act="relu",
):
    from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        max_position_embeddings=max_position_embeddings,
    )
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)

    model = EncoderDecoderModel(config=config)

    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True

    return model


def get_length_from_output(out):
    """
    I use these much slower functions instead of (a != language.PAD_IDX).sum(1)
    because there's an annoying possibility that we sample pad indices.
    """
    batch_size = out.shape[0]
    length = torch.zeros((batch_size,), dtype=torch.int64, device=out.device)
    for i in range(batch_size):
        first_eos = (out[i] == language.EOS_IDX).nonzero()[0]
        length[i] = first_eos + 1
    return length


def get_mask_from_length(length):
    """
    Slow way to get mask from length
    """
    batch_size = length.shape[0]
    max_len = length.max()
    mask = torch.zeros((batch_size, max_len), dtype=torch.uint8, device=length.device)
    for i in range(batch_size):
        this_len = length[i]
        mask[i, :this_len] = 1
    return mask


# Transformer-based encoder decoder model, accepts varying levels of operands
# that are just concatted with special sep token
class OpT(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        # + 1 due to special sep token.
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.sep_idx = vocab_size

        self.seq2seq = get_transformer_encoder_decoder(
            vocab_size + 1,
            embedding_size,
            hidden_size,
            hidden_size,
        )

    def forward(self, inp, y, y_len):
        # Concatenate x1/x2
        if inp:
            inp_seq, inp_len = self.concatenate_input(inp)
        else:
            # Just pass in single tokens for encoder and decoder
            batch_size = y.shape[0]
            inp_seq = torch.ones((batch_size, 1), dtype=y.dtype, device=y.device)
            inp_len = torch.ones((batch_size,), dtype=y.dtype, device=y.device)

        inp_mask = get_mask_from_length(inp_len)
        y_mask = get_mask_from_length(y_len)
        output = self.seq2seq(
            input_ids=inp_seq,
            attention_mask=inp_mask,
            decoder_input_ids=y,
            decoder_attention_mask=y_mask,
            labels=y.clone(),
        )
        decode_len = (y_len - 1).cpu()
        logits = pack_padded_sequence(
            output.logits.cuda(),
            decode_len,
            enforce_sorted=False,
            batch_first=True,
        ).data
        targets = pack_padded_sequence(
            y[:, 1:], decode_len, enforce_sorted=False, batch_first=True
        ).data
        acc = (logits.argmax(1) == targets).float().mean().item()
        return {
            "loss": output.loss,
            "acc": acc,
            "n": logits.shape[0],
        }

    def sample(self, inp, greedy=False, **kwargs):
        if isinstance(inp, int):
            inp_seq = torch.ones((1, 1), dtype=torch.int64, device=self.seq2seq.device)
            inp_len = torch.ones((1,), dtype=torch.int64, device=self.seq2seq.device)
        else:
            inp_seq, inp_len = self.concatenate_input(inp)

        inp_mask = get_mask_from_length(inp_len)
        out_seq = self.seq2seq.generate(
            input_ids=inp_seq,
            attention_mask=inp_mask,
            decoder_start_token_id=language.SOS_IDX,
            pad_token_id=language.PAD_IDX,
            eos_token_id=language.EOS_IDX,
            do_sample=not greedy,
            **kwargs,
        )
        # Assign last value eos, if it wasn't sampled
        out_seq[:, -1] = language.EOS_IDX
        # Compute length
        out_len = get_length_from_output(out_seq)
        scores = None
        return out_seq, out_len, scores

    def concatenate_input(self, inp):
        batch_size = inp[0].shape[0]
        # Remove sos/eos, then add at end
        total_max_len = sum(seq.shape[1] - 2 for seq in inp[::2]) + 2
        # Now add a single sep token for each of the arguments
        n_sep = max(0, len(inp[::2]) - 1)
        total_max_len += n_sep

        total_inp = torch.full(
            (batch_size, total_max_len),
            language.PAD_IDX,
            device=inp[0].device,
            dtype=inp[0].dtype,
        )
        total_len = torch.full(
            (batch_size,),
            language.PAD_IDX,
            device=inp[1].device,
            dtype=inp[1].dtype,
        )

        for i in range(batch_size):
            # For each ith item in the batch, concatenate inputs along all rows

            # Set start of sentence
            total_inp[i, 0] = language.SOS_IDX

            j = 1

            # Track length including sep tokens
            i_len = 0
            for inp_i in range(0, len(inp), 2):
                # For length, ignore sos/eos
                this_inp_seq = inp[inp_i][i]
                this_inp_len = inp[inp_i + 1][i] - 2

                # Ignore sos/eos when copying over sentence
                total_inp[i, j : j + this_inp_len] = this_inp_seq[1 : this_inp_len + 1]
                j += this_inp_len
                i_len += this_inp_len

                # Add sep token
                total_inp[i, j] = self.sep_idx
                j += 1
                i_len += 1

            # Remove the last sep token
            j -= 1
            i_len -= 1
            total_inp[i, j] = language.EOS_IDX

            total_len[i] = i_len + 2

        # Trim
        total_inp = total_inp[:, : total_len.max()]
        return total_inp, total_len


class BinOp(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, fusion="multiply"):
        super().__init__()
        if fusion not in FUSION_MODULES:
            raise NotImplementedError(f"fusion = {fusion}")

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.fusion = FUSION_MODULES[fusion](hidden_size)
        self.encoder = seq2seq.Encoder(vocab_size, embedding_size, hidden_size)
        self.decoder = seq2seq.Decoder(
            vocab_size, hidden_size, embedding_size, hidden_size
        )

    def forward(self, inp, y, y_len):
        """
        fuse(enc(x1), enc(x2)) -> y
        """
        x1, x1_len, x2, x2_len = inp
        x1_enc = self.encoder(x1, x1_len)
        x2_enc = self.encoder(x2, x2_len)

        x_enc = self.fusion(x1_enc, x2_enc)

        return self.decoder(x_enc, y, y_len)

    def sample(self, inp, **kwargs):
        x1, x1_len, x2, x2_len = inp
        x1_enc = self.encoder(x1, x1_len)
        x2_enc = self.encoder(x2, x2_len)

        x_enc = self.fusion(x1_enc, x2_enc)

        return self.decoder.sample(x_enc, **kwargs)


class UnOp(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.encoder = seq2seq.Encoder(vocab_size, embedding_size, hidden_size)
        self.decoder = seq2seq.Decoder(
            vocab_size, hidden_size, embedding_size, hidden_size
        )

    def forward(self, inp, y, y_len):
        """
        enc(x) -> y
        """
        x, x_len = inp
        x_enc = self.encoder(x, x_len)
        return self.decoder(x_enc, y, y_len)

    def sample(self, inp, **kwargs):
        x, x_len = inp
        x_enc = self.encoder(x, x_len)
        return self.decoder.sample(x_enc, **kwargs)


class Primitive(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.decoder = seq2seq.Decoder(
            vocab_size, hidden_size, embedding_size, hidden_size
        )

    def forward(self, inp, y, y_len):
        """
        zeros -> y (ignore input)
        """
        z = torch.zeros(
            (y.shape[0], self.hidden_size),
            dtype=torch.float32,
            device=y.device,
        )
        return self.decoder(z, y, y_len)

    def sample(self, n, **kwargs):
        """
        NOTE: this function has a different signature (accepts batch size since
        it's weird to provide a list of empties)
        """
        z = torch.zeros(
            (n, self.hidden_size),
            dtype=torch.float32,
            device=self.decoder.fc.weight.device,
        )
        return self.decoder.sample(z, **kwargs)


class OpDataset:
    def __init__(
        self,
        optype,
        data,
        models,
        vocab_size,
        greedy_input=False,
        length=1000,
        ignore_missing=True,
        sample=True,
    ):
        self.optype = optype
        self.input = data["in"]
        self.data_size = len(self.input)
        self.models = models
        self.length = length
        self.greedy_input = greedy_input
        self.ignore_missing = ignore_missing
        self.do_sampling = sample

        # Output preprocessing
        # Note OUT already has sos/eos
        self.output_seq, self.output_len = self.to_idx(data["out"])

    def to_idx(self, langs):
        lang_len = np.array([len(t) for t in langs], dtype=np.int)
        lang_idx = np.full((len(langs), max(lang_len)), language.PAD_IDX, dtype=np.int)
        for i, toks in enumerate(langs):
            for j, tok in enumerate(toks):
                lang_idx[i, j] = int(tok)
        return lang_idx, lang_len

    def __len__(self):
        if self.do_sampling:
            return self.length
        else:
            return len(self.input)

    def create_input(self, inp):
        if not inp:
            return []
        seqs = []
        for x in inp:
            arity = len(x) - 1
            if arity == 0:
                # Primitive
                try:
                    primitive_model = self.models[0][x[0]]
                except KeyError:
                    # We have no trained model for this input
                    if self.ignore_missing:
                        return None
                    else:
                        raise RuntimeError(f"No model for {x[0]}")
                *sample, _ = primitive_model.sample(1, greedy=self.greedy_input)
            elif arity == 1:
                op, arg = x
                if len(arg) > 1:
                    raise RuntimeError(f"Found concept with non-primitive arg: {inp}")
                # UnOp. First sample primitive, then apply the transformation
                try:
                    primitive_model = self.models[0][arg[0]]
                except KeyError:
                    # We have no trained model for this input
                    if self.ignore_missing:
                        return None
                    else:
                        raise RuntimeError(f"No model for {arg[0]}")

                *primitive_sample, _ = primitive_model.sample(
                    1, greedy=self.greedy_input
                )

                try:
                    op_model = self.models[1][op]
                except KeyError:
                    if self.ignore_missing:
                        return None
                    else:
                        raise RuntimeError(f"No model for {op}")

                *sample, _ = op_model.sample(primitive_sample, greedy=self.greedy_input)
            else:
                raise NotImplementedError(f"arity {len(x)}")
            seqs.extend(sample)
        # All seqs have batch size 1 - squeeze
        seqs = [s.squeeze(0) for s in seqs]
        return seqs

    def sample(self, i=None):
        if i is None:
            i = np.random.choice(self.data_size)

        out_seq = torch.tensor(self.output_seq[i])
        out_len = self.output_len[i]

        # Construct the input sequence from the concept
        concept = self.input[i]
        # FIXME - this is inefficient
        concept_str = lf_to_concept((self.optype,) + concept)
        inp = self.create_input(concept)
        if inp is None:
            return self.sample()  # Try again

        return (concept_str,) + tuple(inp) + (out_seq, out_len)

    def __getitem__(self, i):
        if self.do_sampling:
            return self.sample()
        else:
            return self.sample(i)


def collect_data(lang, concepts, concepts_split):
    def data_holder():
        return {"in": [], "out": []}

    data = {
        "all": defaultdict(lambda: defaultdict(data_holder)),
        "train": defaultdict(lambda: defaultdict(data_holder)),
        "test": defaultdict(lambda: defaultdict(data_holder)),
    }

    for l, c in zip(lang, concepts):
        arity = len(c) - 1
        # AND, OR, NOT, or a basic feature
        optype = c[0]
        # Input (empty for basic features)
        inp = tuple(c[1:])

        # Add to all
        data["all"][arity][optype]["in"].append(inp)
        data["all"][arity][optype]["out"].append(l)

        # Add to split
        if c in concepts_split["train"]:
            split = "train"
        elif c in concepts_split["test"]:
            split = "test"
        else:
            raise RuntimeError(f"Can't find concept {c}")

        data[split][arity][optype]["in"].append(inp)
        data[split][arity][optype]["out"].append(l)
    return data


def pad_collate_varying(batch):
    batch_flat = list(zip(*batch))
    concepts, *batch_flat = batch_flat

    batch_processed = [concepts]
    for i in range(0, len(batch_flat), 2):
        batch_seq = batch_flat[i]
        batch_len = batch_flat[i + 1]

        batch_len = torch.tensor(batch_len)
        batch_pad = pad_sequence(
            batch_seq, padding_value=language.PAD_IDX, batch_first=True
        )
        batch_processed.extend((batch_pad, batch_len))

    return batch_processed


def train_val_split(opdata, val_pct=0.1, by_concept=False):
    if by_concept:
        concepts = list(set(opdata["in"]))
        # Split by concept
        csize = len(concepts)
        cindices = np.random.permutation(csize)
        val_amt = max(1, int(val_pct * csize))
        val_cidx, train_cidx = cindices[:val_amt], cindices[val_amt:]
        # Now retrieve concepts

        train_concepts = set([c for i, c in enumerate(concepts) if i in train_cidx])
        val_concepts = set([c for i, c in enumerate(concepts) if i in val_cidx])

        train_idx = [i for i, c in enumerate(opdata["in"]) if c in train_concepts]
        val_idx = [i for i, c in enumerate(opdata["in"]) if c in val_concepts]

        assert len(train_idx) + len(val_idx) == len(opdata["in"])
        assert set(train_idx + val_idx) == set(range(len(opdata["in"])))
    else:
        # Just split generically
        dsize = len(opdata["in"])
        assert dsize == len(opdata["out"])
        indices = np.random.permutation(dsize)
        val_amt = int(val_pct * dsize)
        val_idx, train_idx = indices[:val_amt], indices[val_amt:]

    val_idx = set(val_idx)
    train_idx = set(train_idx)

    train_opdata = {
        "in": [x for i, x in enumerate(opdata["in"]) if i in train_idx],
        "out": [x for i, x in enumerate(opdata["out"]) if i in train_idx],
    }
    val_opdata = {
        "in": [x for i, x in enumerate(opdata["in"]) if i in val_idx],
        "out": [x for i, x in enumerate(opdata["out"]) if i in val_idx],
    }
    return train_opdata, val_opdata


def train_model(model, models, optype, opdata, vocab_size, args):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(reduction="none")

    if args.include_not:
        # split by concept if there is more than 1 concept
        by_concept = len(set(opdata["in"])) > 1
    else:
        # split by concept only if and/or (i.e. binary)
        by_concept = optype in {"and", "or"}

    train_opdata, val_opdata = train_val_split(
        opdata, val_pct=0.1, by_concept=by_concept
    )

    train_dataset = OpDataset(optype, train_opdata, models, vocab_size)
    val_dataset = OpDataset(
        optype, val_opdata, models, vocab_size, sample=False, greedy_input=True
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            num_workers=0,
            collate_fn=pad_collate_varying,
            batch_size=args.batch_size,
            shuffle=True,
        ),
        "val": DataLoader(
            val_dataset,
            num_workers=0,
            collate_fn=pad_collate_varying,
            batch_size=args.batch_size,
            shuffle=False,
        ),
    }

    def run_for_one_epoch(split, epoch):
        training = split == "train"
        dataloader = dataloaders[split]
        torch.set_grad_enabled(training)
        model.train(mode=training)

        stats = util.Statistics()
        for batch_i, batch in enumerate(dataloader):
            concepts, *batch = batch

            if args.cuda:
                batch = [x.cuda() for x in batch]
            *inp, out_seq, out_len = batch

            # Preds are from 0 to n-1
            if args.model_type == "transformer":
                output = model(inp, out_seq, out_len)
                loss = output["loss"]
                acc = output["acc"]
                output_batch_size = output["n"]
            else:
                scores, targets = model(inp, out_seq, out_len)

                losses = criterion(scores, targets)
                loss = losses.mean()

                accs = (scores.argmax(1) == targets).float()
                acc = accs.mean()
                output_batch_size = scores.shape[0]

                mbc = compute_metrics_by_concept(
                    concepts,
                    loss=losses.detach().cpu().numpy(),
                    acc=accs.detach().cpu().numpy(),
                )
                for concept, concept_metrics in mbc.items():
                    for metric, cms in concept_metrics.items():
                        n_cm = len(cms)
                        cm_mean = np.mean(cms)
                        stats.update(
                            **{f"{concept}_{metric}": cm_mean}, batch_size=n_cm
                        )

            if not training:
                # TODO - sample and measure top1 accuracy?
                pass

            stats.update(
                **{f"{optype}_loss": loss, f"{optype}_acc": acc},
                batch_size=output_batch_size,
            )

            if training and not args.no_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return stats.averages()

    pbar = tqdm(total=args.epochs)
    best_loss_key = f"best_{optype}_loss"
    best_epoch_key = f"best_{optype}_epoch"
    loss_key = f"{optype}_loss"
    acc_key = f"{optype}_acc"
    metrics = {
        best_loss_key: np.inf,
        best_epoch_key: 0,
    }
    best_model_state = None

    for epoch in range(args.epochs):
        train_metrics = run_for_one_epoch("train", epoch)
        util.update_with_prefix(metrics, train_metrics, "train")

        val_metrics = run_for_one_epoch("val", epoch)
        util.update_with_prefix(metrics, val_metrics, "val")

        if val_metrics[loss_key] < metrics[best_loss_key]:
            metrics[best_loss_key] = val_metrics[loss_key]
            metrics[best_epoch_key] = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        pbar.update(1)
        pbar.set_description(
            f"{epoch} {optype} train loss {train_metrics[loss_key]:3f} train top1 {train_metrics[acc_key]:3f} val loss {val_metrics[loss_key]:3f} val top1 {val_metrics[acc_key]:3f} best loss {metrics[best_loss_key]:3f} @ {metrics[best_epoch_key]}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return metrics


def compute_metrics_by_concept(concepts, **kwargs):
    metrics_by_concept = defaultdict(lambda: defaultdict(list))
    for i, c in enumerate(concepts):
        for metric, vals in kwargs.items():
            metrics_by_concept[c][metric].append(vals[i])
    return metrics_by_concept


def sample(
    split,
    models,
    data,
    vocab_size,
    exp_args,
    args,
    metrics=None,
    epochs=1,
    greedy_input=False,
    greedy=False,
):
    criterion = nn.CrossEntropyLoss(reduction="none")

    if metrics is None:
        metrics = {}

    def sample_from_model(m, optype, opdata):
        dataset = OpDataset(
            optype, opdata, models, vocab_size, greedy_input=greedy_input
        )
        dataloader = DataLoader(
            dataset,
            num_workers=0,
            collate_fn=pad_collate_varying,
            batch_size=args.batch_size,
        )

        stats = util.Statistics()
        samples = []
        for _ in range(epochs):
            for batch_i, batch in enumerate(dataloader):
                concepts, *batch = batch

                if args.cuda:
                    batch = [x.cuda() for x in batch]

                *inp, out_seq, out_len = batch

                with torch.no_grad():
                    if args.model_type == "transformer":
                        output = m(inp, out_seq, out_len)
                        loss = output["loss"]
                        acc = output["acc"]
                        output_batch_size = output["n"]
                    else:
                        scores, targets = m(inp, out_seq, out_len)
                        losses = criterion(scores, targets)
                        loss = losses.mean()
                        output_batch_size = scores.shape[0]
                        accs = (scores.argmax(1) == targets).float()
                        acc = accs.mean()

                        mbc = compute_metrics_by_concept(
                            concepts,
                            loss=losses.detach().cpu().numpy(),
                            acc=accs.detach().cpu().numpy(),
                        )
                        for concept, concept_metrics in mbc.items():
                            for metric, cms in concept_metrics.items():
                                n_cm = len(cms)
                                cm_mean = np.mean(cms)
                                stats.update(
                                    **{f"{concept}_{metric}": cm_mean}, batch_size=n_cm
                                )

                if len(inp) == 0:
                    inp = out_seq.shape[0]  # Specify a number of samples

                lang, lang_len, _ = m.sample(
                    inp, greedy=greedy, max_length=exp_args.max_lang_length
                )
                samples.extend(
                    zip(
                        lang.cpu(),
                        lang_len.cpu(),
                        out_seq.cpu(),
                        out_len.cpu(),
                        concepts,
                    )
                )

                stats.update(loss=loss, acc=acc, batch_size=output_batch_size)

        # Coalesce
        sample_names = [
            "pred_lang",
            "pred_lang_len",
            "model_lang",
            "model_lang_len",
            "gt_lang",
        ]
        samples_arrs = list(zip(*samples))
        samples = {name: samples_arrs[i] for i, name in enumerate(sample_names)}

        return stats.averages(), samples

    samples = defaultdict(list)
    pbar = tqdm(total=sum(len(x) for arity, x in data.items() if arity in data))

    for arity in (0, 1, 2):
        if arity not in data:
            continue
        adata = data[arity]
        for optype, opdata in adata.items():
            pbar.set_description(f"Sample: {optype}")
            # Get the model
            try:
                m = models[arity][optype]
            except KeyError:
                raise RuntimeError(f"No model for {optype}")
            m.eval()
            if args.cuda:
                m.cuda()

            op_metrics, op_samples = sample_from_model(m, optype, opdata)
            util.update_with_prefix(metrics, op_metrics, split)

            # Save the model
            for name, vals in op_samples.items():
                samples[name].extend(vals)

            pbar.update(1)

    pbar.close()
    return metrics, dict(samples)


def get_model(arity, vocab_size, args):
    if args.model_type == "rnn":
        if arity == 0:
            m = Primitive(vocab_size, args.embedding_size, args.hidden_size)
        elif arity == 1:
            m = UnOp(vocab_size, args.embedding_size, args.hidden_size)
        elif arity == 2:
            m = BinOp(
                vocab_size,
                args.embedding_size,
                args.hidden_size,
                fusion="multiply",
            )
    else:
        # Generic transformer model
        m = OpT(vocab_size, args.embedding_size, args.hidden_size)
    return m


def train(data, vocab_size, args):
    models = defaultdict(dict)
    metrics = {}

    pbar = tqdm(total=sum(len(x) for x in data.values()))
    for arity in (0, 1, 2):
        adata = data[arity]
        for optype, opdata in adata.items():
            pbar.set_description(f"Train: {optype}")

            m = get_model(arity, vocab_size, args)

            if args.cuda:
                m = m.cuda()

            op_metrics = train_model(m, models, optype, opdata, vocab_size, args)
            util.update_with_prefix(metrics, op_metrics, "train")

            # Save the model
            models[arity][optype] = m

            pbar.update(1)

    pbar.close()
    return models, metrics


TOK_NAMES = string.ascii_lowercase + "".join(map(str, range(10)))


def anonymize(out):
    out_anon = []
    for tok in out:
        i = int(tok)
        try:
            tok_anon = TOK_NAMES[i]
        except IndexError:
            tok_anon = str(i + 10)
        out_anon.append(tok_anon)
    return out_anon


def flatten(nested):
    return tuple(_flatten(nested))


def _flatten(nested):
    if isinstance(nested, str):
        return [
            nested,
        ]
    else:
        flattened = []
        for item in nested:
            flattened.extend(_flatten(item))
        return flattened


def flatten_opdata(opdata):
    concepts = []
    messages = []
    for concept, cdata in opdata.items():
        for m in cdata:
            concepts.append(concept)
            messages.append(m)
    return concepts, messages


def get_opname(concept_flat):
    if concept_flat[0] == "and":
        return "AND"
    elif concept_flat[0] == "or":
        return "OR"
    elif concept_flat[0] == "not":
        return "NOT"
    else:
        return "prim"


def get_data_stats(data, unique_concepts):
    records = []
    entropies = []
    concepts = []
    messages = []
    for arity, adata in data.items():
        for optype, opdata in adata.items():

            # Sort language further by concept (i.e. op + args)
            opdata_by_concept = defaultdict(list)
            for inp, out in zip(opdata["in"], opdata["out"]):
                out = " ".join(anonymize(out))
                opdata_by_concept[(optype,) + inp].append(out)

            # Flatten + extend
            a_cs, a_ms = flatten_opdata(opdata_by_concept)
            concepts.extend(a_cs)
            messages.extend(a_ms)

            # For each concept, get distribution of utterances
            for concept, cdata in opdata_by_concept.items():
                counts = Counter(cdata)
                counts_total = sum(counts.values())
                counts_norm = {k: v / counts_total for k, v in counts.items()}

                entropy = scipy.stats.entropy(list(counts.values()))
                entropies.append(entropy)

                if concept in unique_concepts["train"]:
                    seen = "seen"
                elif concept in unique_concepts["test"]:
                    seen = "unseen"
                else:
                    raise RuntimeError(f"Can't find concept {concept}")

                concept_flat = flatten(concept)
                concept_str = " ".join(concept_flat)

                ctype = get_opname(concept_flat)

                for lang, count in counts.items():
                    percent = counts_norm[lang]
                    concept_flat = " ".join(flatten(concept))
                    records.append(
                        {
                            "arity": arity,
                            "type": ctype,
                            "concept": concept_str,
                            "lang": lang,
                            "count": count,
                            "percent": percent,
                            "entropy": entropy,
                            "seen": seen,
                        }
                    )

    concept_df = pd.DataFrame(records)

    # Overall stats - (1) MI; (2) conditional entropy
    concepts = [" ".join(flatten(c)) for c in concepts]
    mi = mutual_info_score(concepts, messages)
    ami = adjusted_mutual_info_score(concepts, messages)
    overall_stats = {
        "entropy": np.mean(entropies),
        "mi": mi,
        "ami": ami,
    }
    return concept_df, overall_stats


def metrics_to_df(metrics):
    records = []
    for mname, value in metrics.items():
        split, *optional_concept, metric = mname.split("_")
        if optional_concept:
            concept = optional_concept[0]
            try:
                concept_lf = concept_to_lf(concept)
            except AssertionError:
                continue
            arity = len(concept_lf) - 1

            concept_flat = flatten(concept_lf)
            concept = " ".join(concept_flat)
            op = get_opname(concept_flat)
        else:
            concept = "overall"
            arity = float("nan")
            op = "overall"
        records.append(
            {
                "split": split,
                "concept": concept,
                "metric": metric,
                "value": value,
                "arity": arity,
                "op": op,
            }
        )
    return pd.DataFrame(records)


def split_higher_order_concepts(concepts, test_percent=0.1, seed=None):
    if seed is not None:
        random.seed(seed)
    ctypes = [
        [v for v in concepts if v[0] == "and"],
        [v for v in concepts if v[0] == "not"],
        [v for v in concepts if v[0] == "or"],
    ]
    train_concepts = []
    test_concepts = []
    for ctype in ctypes:
        n_test_concepts = int(0.1 * len(ctype))
        ctype = sorted(ctype)
        random.shuffle(ctype)
        train_concepts.extend(ctype[n_test_concepts:])
        test_concepts.extend(ctype[:n_test_concepts])
    return train_concepts, test_concepts


def anonymize_true_lang(tl):
    tl_anon = []
    anon_vocab = {"pad": 0, "sos": 1, "eos": 2}  # these values not used
    for lang in tl:
        lang_anon = ["1"]
        for tok in lang:
            if tok not in anon_vocab:
                anon_vocab[tok] = len(anon_vocab)
            lang_anon.append(str(anon_vocab[tok]))
        lang_anon.append("2")
        tl_anon.append(lang_anon)
    return tl_anon


def process(lang_file, args):
    exp_dir = os.path.split(lang_file)[0]
    exp_args = Namespace(**util.load_args(exp_dir))

    lf = pd.read_csv(lang_file, keep_default_na=False)

    if args.true_lang:
        lang = lf["true_lang"]
        # Turn true lang into something that looks like lang
        lang = [x.strip().split(" ") for x in lang]
        lang = anonymize_true_lang(lang)
    else:
        lang = lf["lang"]
        lang = [x.strip().split(" ") for x in lang]
        # Empty strings should be no tokens
        lang = [[] if x == [""] else x for x in lang]

    if "shapeworld" in args.dataset:
        concepts = [concept_to_lf(x) for x in lf["true_lang"]]
        # Load worlds, get concept train/test split
        unique_concepts = get_unique_concepts(args.dataset)
        unique_concepts = {
            k: list(map(concept_to_lf, vs)) for k, vs in unique_concepts.items()
        }
        if not args.standard_split:
            all_concepts = unique_concepts["train"] + unique_concepts["test"]
            # XXX: these could include duplicates (e.g. if there's no config
            # split in the dataset), so make this a set.
            all_concepts = list(set(all_concepts))
            if args.include_not:
                higher_threshold = 1
            else:
                higher_threshold = 2
            primitive_concepts = [v for v in all_concepts if len(v) <= higher_threshold]
            higher_order_concepts = [
                v for v in all_concepts if len(v) > higher_threshold
            ]
            train_h_concepts, test_h_concepts = split_higher_order_concepts(
                higher_order_concepts,
                seed=0,
            )
            unique_concepts = {
                "train": primitive_concepts + train_h_concepts,
                "test": test_h_concepts,
            }
        if (
            "force_ref" not in lang_file
            and "force_concept" not in lang_file
            and "force_setref" not in lang_file
        ):
            with open(os.path.join(exp_dir, "acre_split.json"), "w") as f:
                json.dump(unique_concepts, f)
    else:
        concepts = [(str(x),) for x in lf["true_lang"]]
        unique_concepts = {
            # FIXME - this is slightly off since CUB are 1-indexed.
            "train": [(str(i),) for i in range(1, 100)],
            "test": [(str(i),) for i in range(150, 200)],
        }

    data = collect_data(lang, concepts, unique_concepts)

    data_stats, overall_stats = get_data_stats(data["all"], unique_concepts)
    data_stats.to_csv(lang_file.replace(".csv", "_stats.csv"), index=False)
    with open(lang_file.replace(".csv", "_overall_stats.json"), "w") as f:
        json.dump(overall_stats, f)

    vocab_size = exp_args.vocab_size + 3
    if not args.stats_only:
        # Train on train split
        models, _ = train(data["train"], vocab_size, args)
        # Eval + sample language on both train and test splits
        for split in ["train", "test"]:
            split_metrics, split_lang = sample(
                split,
                models,
                data[split],
                vocab_size,
                exp_args,
                args,
                epochs=args.sample_epochs,
                greedy=args.greedy,
                greedy_input=args.greedy_input,
            )
            # Save to metrics
            split_metrics_df = metrics_to_df(split_metrics)
            split_metrics_df.to_csv(
                lang_file.replace(".csv", f"_{split}_acre_metrics.csv"), index=False
            )
            torch.save(
                split_lang, lang_file.replace(".csv", f"_{split}_sampled_lang.pt")
            )


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("lang_files", nargs="+")
    parser.add_argument("--model_type", choices=["rnn", "transformer"], default="rnn")
    parser.add_argument("--dataset", default="./data/shapeworld")
    parser.add_argument("--embedding_size", default=50, type=int)
    parser.add_argument(
        "--standard_split", action="store_true", help="Use default train/test split"
    )
    parser.add_argument("--hidden_size", default=100, type=int)
    parser.add_argument("--true_lang", action="store_true", help="Use true language")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--sample_epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stats_only", action="store_true")
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument(
        "--greedy", action="store_true", help="Greedy decoding from ACRe model"
    )
    parser.add_argument(
        "--include_not",
        action="store_true",
        help="Include NOT operators (by default excluded since transformer doesn't generalize across NOT even w/ true_lang",
    )
    parser.add_argument(
        "--greedy_input",
        action="store_true",
        help="Greedy decoding from models that constitute ACRe model",
    )

    args = parser.parse_args()

    for lang_file in tqdm(args.lang_files):
        process(lang_file, args)

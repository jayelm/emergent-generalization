import numpy as np
import os
import gzip
import h5py

import torch
import torch.nn.functional as F

try:
    import ujson as json
except ImportError:
    import json

from . import generic
from . import language
from . import util


SHAPES = ["circle", "ellipse", "square", "rectangle", "triangle"]
COLORS = ["red", "blue", "green", "yellow", "white", "gray"]
SHAPES_DICT = {v: k for k, v in enumerate(SHAPES)}
COLORS_DICT = {v: k for k, v in enumerate(COLORS)}
N_FEATS = len(SHAPES) + len(COLORS)


SPLITS = ["train", "val", "val_same", "test", "test_same"]


def get_unique_concepts(dfolder):
    split_langs = {}

    for split in ["train", "val", "test"]:
        data_file = os.path.join(dfolder, f"{split}.npz")
        if os.path.exists(data_file):
            data = np.load(data_file)
        else:
            # Try hdf5
            data = h5py.File(data_file.replace(".npz", ".hdf5"), "r")

        if data["langs"].dtype != np.unicode:
            langs_decoded = [lang.decode("utf-8") for lang in data["langs"]]
        else:
            langs_decoded = data["langs"]

        split_langs[split] = langs_decoded

    return {
        "train": set(list(split_langs["train"])),
        "test": set(list(split_langs["val"]) + list(split_langs["test"])),
    }


def _concept_to_lf(concept):
    """
    Parse concept. Since fixed recursion we don't have to worry about
    precedence
    """
    op = ""
    if "or" in concept:
        op = "or"
    elif "and" in concept:
        op = "and"

    if op:
        op_index = concept.index(op)
        left = concept[:op_index]
        right = concept[op_index + 1 :]
        return (op, _concept_to_lf(left), _concept_to_lf(right))

    if "not" in concept:
        assert len(concept) == 2, f"unable to parse {concept}"
        return ("not", (concept[1],))

    assert len(concept) == 1, f"unable to parse {concept}"
    return (concept[0],)


def concept_to_lf(concept, split=True):
    if split:
        concept = concept.split(" ")
    return _concept_to_lf(concept)


def lf_to_concept(lf):
    if isinstance(lf, str):
        return lf
    if len(lf) == 1:  # Primitive
        return lf_to_concept(lf[0])
    elif len(lf) == 2:
        assert lf[0] == "not"
        arg_concept = lf_to_concept(lf[1])
        return f"{lf[0]} {arg_concept}"
    elif len(lf) == 3:
        l_concept = lf_to_concept(lf[1])
        r_concept = lf_to_concept(lf[2])
        return f"{l_concept} {lf[0]} {r_concept}"


def extract_shapes(worlds):
    """
    Turn worlds into one-hot arrays.
    """
    shapes = []

    for world in worlds:
        imgs = world["imgs"]
        this_world_shapes = []

        if len(imgs[0]) > 1:
            raise RuntimeError("More than one shape in this world")
        for img in imgs:
            shape = img[0]
            this_world_shapes.append([shape["color"], "and", shape["shape"]])

        shapes.append(this_world_shapes)

    return shapes


def extract_concepts(worlds):
    concepts = {}
    for game in worlds:
        concept = game["config"]["concept"]
        assns = game["config"]["pos"]
        if concept not in concepts:
            concepts[concept] = assns
    return concepts


def concepts_to_onehot(concepts):
    # Assign unique one hot values
    w2i = {}
    for assns in concepts.values():
        for assn in assns:
            for tok in assn:
                if tok not in w2i:
                    w2i[tok] = len(w2i)
    concepts_onehot = {}
    for concept, assns in concepts.items():
        assns_onehot = []
        for assn in assns:
            assn_onehot = np.zeros(len(w2i), dtype=np.uint8)
            for tok in assn:
                assn_onehot[w2i[tok]] = 1
            assns_onehot.append(assn_onehot)
        concepts_onehot[concept] = np.stack(assns_onehot)
    return concepts_onehot


def load_other_data(this_game_type, split, dataset, fast=False, into_memory=False):
    """Load the opposite of the current game's dataset."""
    if this_game_type == "ref":
        # Load concept data
        assert "_ref" in dataset
        other_dataset = dataset.replace("_ref", "")
    else:
        other_dataset = "data/shapeworld_ref/"
    return load_split(other_dataset, split, fast=fast, into_memory=into_memory)


def load(args, fast=False):
    datas = {}
    if args.backbone == "resnet18":
        # Need larger images
        image_size = 224
    else:
        image_size = 64

    for split in SPLITS:
        sfile = os.path.join(args.dataset, f"{split}.npz")
        sfile_hdf5 = sfile.replace(".npz", ".hdf5")
        is_present = os.path.exists(sfile) or os.path.exists(sfile_hdf5)
        if not is_present:
            if not split.endswith("_same"):
                # Then this split should be here
                raise RuntimeError(f"Can't find {sfile} or {sfile_hdf5}")
            else:
                continue
        datas[split] = load_split(
            args.dataset, split, fast=fast, into_memory=args.load_shapeworld_into_memory
        )

    langs = np.concatenate([datas[s]["langs"] for s in datas])
    vocab = language.init_vocab(langs)

    # Compute vocab first
    _, md_vocab = get_metadata(langs)

    # Combine concepts
    if fast:
        concepts = None
        concept_distances = None
    else:
        concepts = {}
        for split in datas:
            concepts.update(datas[split]["concepts"])
        concepts = concepts_to_onehot(concepts)
        concept_distances = util.get_pairwise_hausdorff_distances(concepts)

    dataset_kwargs = {
        "n_examples": args.n_examples,
        "visfunc": generic.vis_image,
        "name": "shapeworld",
        "image_size": image_size,
    }

    datasets = {}
    for split in datas:
        datas[split]["metadata"] = get_metadata(datas[split]["langs"], md_vocab)[0]
        datasets[split] = ShapeWorldDataset(
            datas[split],
            vocab,
            augment=split == "train",
            percent_novel=args.percent_novel,
            reference_game=args.reference_game,
            shapes=datas[split]["shapes"],
            concepts=concepts,
            concept_distances=concept_distances,
            metadata_vocab=md_vocab,
            **dataset_kwargs,
        )

    # Load other versions of datasets for eval
    this_game_type = util.get_game_type(args)
    for _split in ["val", "test", "val_same", "test_same"]:
        # Load the other dataset
        other_data = load_other_data(
            this_game_type,
            _split,
            args.dataset,
            fast=fast,
            into_memory=args.load_shapeworld_into_memory,
        )
        other_data["metadata"] = get_metadata(other_data["langs"], md_vocab)[0]
        # other_vocab is used to measure the correct concept distances
        other_vocab = language.init_vocab(other_data["langs"])
        # Load other concepts
        if fast:
            other_concepts = None
            other_concept_distances = None
        else:
            other_concepts = concepts_to_onehot(other_data["concepts"])
            other_concept_distances = util.get_pairwise_hausdorff_distances(
                other_concepts
            )

        for game_type in ["ref", "setref", "concept"]:
            split = f"{_split}_{game_type}"
            if game_type == this_game_type:
                datasets[split] = datasets[_split]
            else:
                datasets[split] = ShapeWorldDataset(
                    other_data,
                    other_vocab,
                    augment=False,
                    percent_novel=1.0 if game_type == "concept" else 0.0,
                    reference_game=game_type == "ref",
                    shapes=other_data["shapes"],
                    concepts=other_concepts,
                    concept_distances=other_concept_distances,
                    metadata_vocab=md_vocab,
                    **dataset_kwargs,
                )

    return datasets


def load_split(dataset, split, fast=False, into_memory=False):
    data_file = os.path.join(dataset, f"{split}.npz")
    if os.path.exists(data_file):
        data = np.load(data_file)
    else:
        # Try hdf5
        data = h5py.File(data_file.replace(".npz", ".hdf5"), "r")
    # Load shapes for reference games
    if fast:
        shapes = None
        concepts = None
    else:
        world_file = os.path.join(dataset, f"{split}_worlds.json")
        if os.path.exists(world_file):
            with open(world_file, "r") as f:
                worlds = json.load(f)
        else:
            with gzip.open(world_file + ".gz", "r") as f:
                worlds = json.load(f)
        shapes = extract_shapes(worlds)
        concepts = extract_concepts(worlds)

    imgs = data["imgs"]
    labels = data["labels"]
    if into_memory:
        imgs = imgs[:]
        labels = labels[:]

    if data["langs"].dtype != np.unicode:
        langs_decoded = [lang.decode("utf-8") for lang in data["langs"]]
    else:
        langs_decoded = data["langs"]

    # Force 1D object array
    langs = np.empty(len(langs_decoded), dtype=np.object)
    langs[:] = [t.lower().split() for t in langs_decoded]

    return {
        "x": imgs,
        "labels": labels,
        "langs": langs,
        "shapes": shapes,
        "concepts": concepts,
    }


def feature_type(feat):
    if feat in COLORS:
        return "color"
    elif feat in SHAPES:
        return "shape"
    else:
        raise ValueError(f"Unknown feature type {feat}")


def get_metadata(langs, md_vocab=None):
    md = []
    if md_vocab is None:
        md_vocab = {
            "w2i": {},
            "i2w": {},
        }
    for lang in langs:
        lc = concept_to_lf(lang, split=False)
        if len(lc) == 1:
            # Single feature. Ignore NOTs (treat them the same)
            this_md = feature_type(lc[0])
            pass
        elif len(lc) == 2:
            # NOT
            this_md = feature_type(lc[1][0])
            pass
        elif len(lc) == 3:
            op = lc[0]
            if len(lc[1]) == 1:
                l_md = feature_type(lc[1][0])
            else:
                l_md = feature_type(lc[1][1][0])
            if len(lc[2]) == 1:
                r_md = feature_type(lc[2][0])
            else:
                r_md = feature_type(lc[2][1][0])
            this_md = f"{op}_{l_md}_{r_md}"
        else:
            raise ValueError(f"Unknown feature type {this_md}")
        if this_md not in md_vocab["w2i"]:
            md_i = len(md_vocab["w2i"])
            md_vocab["w2i"][this_md] = md_i
            md_vocab["i2w"][md_i] = this_md
        else:
            md_i = md_vocab["w2i"][this_md]

        md.append(md_i)
    return md, md_vocab


class ShapeWorldDataset(generic.ConceptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Tokenize metadata language
        if self.shapes is not None:
            self.shape_lang_idx, self.shape_lang_len = self.shapes_to_idx()

    @util.return_index
    def get_reference_game(self, i):
        # FIXME - code reuse here (maybe use a temporary lang?)
        img = self.x[i]
        label = self.labels[i]
        md = self.metadata[i]

        midp = img.shape[0] // 2

        # Choose a single random target
        if self.augment:
            pos_i = np.random.randint(midp)
        else:
            pos_i = 0

        # lang to be the shape of the positive target
        lang = self.shape_lang_idx[i, pos_i]
        # Re-assign positive examples
        img[:midp] = img[pos_i]

        if self.augment:
            # Shuffle positives by themselves
            pos_order = np.random.permutation(midp)
            img[:midp] = img[:midp][pos_order]
            # Shuffle negatives by themselves
            neg_order = np.random.permutation(midp)
            img[midp:] = img[midp:][neg_order]

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        if self.image_size is not None and self.image_size != img.shape[2]:
            img = F.interpolate(img, (self.image_size, self.image_size))

        splits = util.split_spk_lis(img, label, self.n_examples, percent_novel=0.0)
        return splits + (lang, md)

    def __getitem__(self, i):
        if self.reference_game:
            return self.get_reference_game(i)
        else:
            return super().__getitem__(i)

    def shapes_to_idx(self):
        n = len(self.shapes)
        n_img = len(self.shapes[0])
        shape_lang_len = np.full((n, n_img), 5, dtype=np.int)
        shape_lang_idx = np.zeros((n, n_img, 5), dtype=np.int)
        for i in range(n):
            for j in range(n_img):
                shape_lang_idx[i, j, 0] = self.w2i[language.SOS_TOKEN]
                for tok_i, tok in enumerate(self.shapes[i][j], start=1):
                    shape_lang_idx[i, j, tok_i] = self.w2i.get(
                        tok, self.w2i[language.UNK_TOKEN]
                    )
                shape_lang_idx[i, j, -1] = self.w2i[language.EOS_TOKEN]
        return shape_lang_idx, shape_lang_len

    def concept_distance(self, c1, c2):
        pair = tuple(sorted((c1, c2)))
        return self.concept_distances[pair]

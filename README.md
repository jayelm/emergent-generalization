# Emergent Communication of Generalizations

https://arxiv.org/abs/2106.02668

NeurIPS 2021

## Setup

- Download and process birds (CUB) data [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), then unzip into `/data/cub` directory (i.e. the filepath should be `data/cub/CUB_200_2011/*`), then run `python save_cub_np.py` in the `/data` directory to save cub images to easily accessible npz format.
- Download and process ShapeWorld data: use `data/download_shapeworld.sh`. There are 3 datasets:
    - [shapeworld](http://nlp.stanford.edu/data/muj/emergent-generalization/shapeworld/shapeworld.tar.gz): 20k games over the 312 conjunctive concepts
    - [shapeworld_ref](http://nlp.stanford.edu/data/muj/emergent-generalization/shapeworld/shapeworld_ref.tar.gz): 20k games over the 30 conjunctive concepts possible for
        reference games only
    - [shapeworld_all](http://nlp.stanford.edu/data/muj/emergent-generalization/shapeworld/shapeworld_all.tar.gz): 20k games over the 312 conjunctive concepts, *but no
        compositional split*.

Note that running concept/setref game agents loads the shapeworld ref dataset
(and vice versa) because we do zero shot eval, so download everything.

Code used for generating the ShapeWorld data is located [here](https://github.com/jayelm/minishapeworld/tree/neurips2021).

## Running experiments

### Quickstart

`./run_cub.sh` and `./run_sw.sh` contain ready-to-go commands for running
reference, set reference, and concept agents for birds and ShapeWorld,
respectively. See those scripts for more details. If you need more info, read
on for the basic workflow.

### 1. Train model

Run `python code/train.py --cuda --name NAME --dataset DATASET` where `NAME` is
an experiment name (results saved in `exp/NAME`) and `DATASET` is the dataset
(`data/cub`, `data/shapeworld`, or `data/shapeworld_ref`).
To specify the type of game use the following additional flags:

- `--percent_novel 1.0`: runs concept game (i.e. percent novel indicates what
    % of images are novel to the student; note you can try values in between if
    you'd like)
- `--percent_novel 0.0`: runs setref game
- `--percent_novel 0.0 --reference_game`: runs reference game. **For
    ShapeWorld, be sure to pass in the 30-concept reference game dataset `shapeworld_ref`,
    not the standard 312-concept dataset `shapeworld`!**

Additional flags relevant for experiments:
- `--max_lang_length`: maximum message length. **This includes sos/eos token,
    so the true length is this value minus 2**
- `--vocab_size`: vocab size of the agents.
- `--n_examples`: number of examples given to agents
- `--uniform_weight`: add uniform noise to gumbel softmax exploration policy
- `--wandb`: activate wandb logging (run `wandb init` yourself)

There are other options documented in `code/io_util.py`.


#### Metrics

Metrics are logged into `exp/NAME/metrics.csv` and logged to wandb (if
`--wandb` is enabled). The relevant ones are:

- `train_acc`:
- `test`. For shapeworld, this metric is split into `{test,val}_acc` and `{test,val}_same_acc` to denote unseen and seen splits, respectively, where `{test,val}_avg_acc` averages the two.
- `{train,test}_langts`/`{train,test}_ts`: edit-distance based topographic similarity. For Birds (`cub`), the metric is `ts`; for ShapeWorld the metric is `langts`.
- `{train,test}_hausdorff`: Hausdorff distance based topographic similarity.

There are many other metrics, most of which should be reasonably intuitive,
though contact authors for clarifications.

There are also all of the above metrics split by game type (we eval ref agents on setref, concept, etc).

Mutual information and entropy are measured later (see below).

### 2. Sample language from model

The above command produces a `metrics.csv` with most metrics, but I measure
entropy and AMI at the end by sampling a bunch of language from the model and
analyzing that corpus. To do so, run

```
# (no --cuda flag needed; will use whatever flag was set at train time)
python code/sample.py exp/NAME
```

which by default samples 200k messages from a trained model into
`exp/NAME/sampled_lang.csv` and some summary statistics into
`exp/NAME/sampled_stats.json`.

Now, if you just want the information theoretic systematicity metrics, for both
Birds and ShapeWorld run
`python code/acre.py exp/NAME/sampled_lang.csv --dataset DATASET --cuda --stats_only`
which **does not run ACRe**, but rather just dumps some summary statistics:

- `exp/NAME/sampled_lang_overall_stats.json`: this contains entropy,
    unnormalized mutual information, and adjusted mutual information
- `exp/NAME/sampled_lang_stats.csv`: this is a list of utterances generated for
    each concept, with their counts. Also entropy information. This can be used
    to plot the sunburst (i.e. nested pie) plots in the paper. See "4.
    Visualizing Model Outputs"

Again, we haven't actually run ACRe. If you want to run ACRe, read on:

### 3. Train ACRe

If you actually want to train an ACRe model you should train your model with
the `shapeworld_all` dataset, which doesn't involve the compositional split
(though you can still do ACRe analysis on models trained normally).

Run ACRe without the `--stats_only` flag. Rather, run

`python code/acre.py exp/NAME/sampled_lang.csv --dataset DATASET --cuda`

which trains an ACRe model to reconstruct the agent language according to the
concepts of `DATASET`. This prints out some top1 acc/loss metrics and the
following files:

- `exp/NAME/sampled_lang_{train,test}_acre_metrics.csv`: overall loss/top1 acc
    for ACRe reconstruction compared to the ground truth language only (i.e.
    not evaluating a listener model yet), as well as these metrics broken down
    by concept
- `exp/NAME/sampled_lang_{train,test}_sampled_lang.pt`: Contains ground truth
    model language for both train/test ACRe splits, as well as ACRe
    reconstructions. This gets used to evaluate a listener in the next section.
- `exp/NAME/acre_split.json`: The split of train/test concepts used for ACRe.

### 4. Evaluate ACRe on Listener

Run

`python code/eval_zero_shot.py exp/NAME --cuda`.

which evaluates across `--epochs` epochs (default 5), categorizing concepts by
whether they belong to the ACRe train or test split, and evaluates several
types of language on the listener:

- `ground_truth_1`: the model lang located in `exp/NAME/sampled_lang_{train,test}_sampled_lang.pt`.
- `same_concept`: language sampled from other model utterances from the same concept
- `acre`: ACRe reconstructed language.
- `random`: random language uniformly sampled from the possible set of
    utterances (not reported in paper; worse than `any_concept`)
- `any_concept`: random language sampled from utterances from any concept (the random baseline in the paper)
- `closest_concept`: language sampled from utterances for the "closest" concept as measured by edit distance
- `ground_truth_2`: (sanity check) re-sample language from the teacher; should be close to `ground_truth_1` performance.

These results are saved into

- `exp/NAME/zero_shot_{train,test}.json`: BLEU-1 and listener acc aggregated
    across all concepts, and for each concept individually
- `exp/NAME/zero_shot_lang_type_stats.csv`: a lang stats file similar to
    `exp/NAME/sampled_lang_stats.csv` described above, which can be used to
    visualize outputs for the various language distributions as described in
    the next section.

### 5. Visualizing model outputs

This requires `R` and the `sunburstR` package, as well as a generated
`sampled_lang_stats.csv` which is produced by `acre.py` (just the
`--stats_only` flag will do). Then an example usage is located in lines
425--441 of `analysis/analysis.Rmd`.

### 6. Evaluating across different games

Accuracy and topographic similarity metrics are evaluated zero-shot across
different games in the main train script, though entropy/AMI metrics aren't
collected. To obtain those, and to get all the results in one place, sample
language while using a `--force_*` flag to force the game to be ref, setref, or
concept. This adds a `_force_{ref,setref,concept}` prefix to every file
outputted by `sample.py`, e.g. `sampled_lang_force_ref.csv`. For example:

```
python code/sample.py exp/NAME --force_reference_game
python code/acre.py exp/NAME/sampled_lang_force_ref.csv --dataset data/shapeworld_ref
```

which now produces `exp/NAME/sampled_stats_force_ref.json`,
`exp/NAME/sampled_lang_force_ref_overall_stats.json`,
`exp/NAME/sampled_lang_force_ref_stats.csv`, etc.

**If you're analyzing ShapeWorld, remember to specify the right dataset - either
ref, or setref/concept - when printing summary statistics via `acre.py`**.

## Dependencies

This code was tested with python 3.8 and `torch==1.8.1`. A specific
environments file is located in `requirements.txt`, but other common package
versions are likely to be compatible as well.

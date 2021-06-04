#!/bin/bash

MAXLEN="10"  # max message length. note this includes sos/eos token, i.e. the true length is this - 2
TOK="20"  # vocab size of the agents
N_WORKERS="4"
EX="10"  # number of examples given to teacher/student, divided evenly b/t targets and distractors (must be even!)
REF_DATASET="data/cub" # path to reference game dataset (for cub, it's the same thing)
DATASET="data/cub"  # path to standard dataset
BACKBONE="resnet18"  # vision backbone (imagenet pretrained: set --pretrained_feat_model)
BATCH_SIZE="16"  # true batch size is BATCH_SIZE x ACCUM_STEPS
ACCUM_STEPS="1"
EPOCHS="100"  # number of epochs to train for

if [ -z "$1" ]; then
    echo "usage: ./run_cub.sh EXP_NAME"
    exit 1
fi
i="$1"  # specify a name/tag of some sort to track this run

# ==== CONCEPT ====
# percent_novel = 1.0, reference_game = False
EXP_NAME="cub_concept_""$EX""_ex_$i"

python code/train.py --cuda --name "$EXP_NAME" --n_examples $EX --dataset $DATASET --percent_novel 1.0 --max_lang_length $MAXLEN --vocab_size $TOK --batch_size $BATCH_SIZE --accum_steps $ACCUM_STEPS --n_workers $N_WORKERS --backbone $BACKBONE --epochs $EPOCHS --pretrained_feat_model

python code/sample.py exp/$EXP_NAME
# NOTE: for CUB, this doesn't actually run ACRe, just produces summary statistics (including entropy/AMI measures) in sampled_stats.json
python code/acre.py exp/$EXP_NAME/sampled_lang.csv --stats_only --dataset $DATASET --cuda

# Example of zero shot eval on other game types. If you need these to run faster specify less number of samples with the --n flag to sample.py
# python code/sample.py exp/$EXP_NAME --force_reference_game
# python code/acre.py exp/$EXP_NAME/sampled_lang_force_ref.csv --stats_only --dataset $DATASET --cuda

# python code/sample.py exp/$EXP_NAME --force_setref_game
# python code/acre.py exp/$EXP_NAME/sampled_lang_force_setref.csv --stats_only --dataset $DATASET --cuda

# ==== SETREF ====
# percent_novel = 0.0, reference_game = False
EXP_NAME="cub_setref_""$EX""_ex_$i"

python code/train.py --cuda --name "$EXP_NAME" --n_examples $EX --dataset $DATASET --percent_novel 0.0 --max_lang_length $MAXLEN --vocab_size $TOK --batch_size $BATCH_SIZE --accum_steps $ACCUM_STEPS --n_workers $N_WORKERS --backbone $BACKBONE --epochs $EPOCHS --pretrained_feat_model

python code/sample.py exp/$EXP_NAME
# NOTE: for CUB, this doesn't actually run ACRe, just produces summary statistics (including entropy/AMI measures) in sampled_stats.json
python code/acre.py exp/$EXP_NAME/sampled_lang.csv --stats_only --dataset $DATASET --cuda

# ==== REF ====
# percent_novel = 0.0, reference_game = True
EXP_NAME="cub_ref_""$EX""_ex_$i"

python code/train.py --cuda --name "$EXP_NAME" --n_examples $EX --dataset $REF_DATASET --percent_novel 0.0 --reference_game --max_lang_length $MAXLEN --vocab_size $TOK --batch_size $BATCH_SIZE --accum_steps $ACCUM_STEPS --n_workers $N_WORKERS --backbone $BACKBONE --epochs $EPOCHS --pretrained_feat_model

python code/sample.py exp/$EXP_NAME
# NOTE: for CUB, this doesn't actually run ACRe, just produces summary statistics (including entropy/AMI measures) in sampled_stats.json
python code/acre.py exp/$EXP_NAME/sampled_lang.csv --stats_only --dataset $DATASET --cuda

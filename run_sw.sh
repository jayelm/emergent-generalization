#!/bin/bash

MAXLEN="7"  # max message length. note this includes sos/eos token, i.e. the true length is this - 2
TOK="14"  # vocab size of the agents
N_WORKERS="0"
EX="20"  # number of examples given to teacher/student, divided evenly b/t targets and distractors (must be even!)
REF_DATASET="data/shapeworld_ref"  # path to reference game dataset (shapeworld: subset, only 30 conjunctive concepts)
DATASET="data/shapeworld"  # path to standard dataset
BACKBONE="conv4"  # vision backbone
BATCH_SIZE="32"  # true batch size is BATCH_SIZE x ACCUM_STEPS
ACCUM_STEPS="4"
UNIF="0.1"  # uniform weight for exploration
EPOCHS="100"  # number of epochs to train for


# Note: --load_shapeworld_into_memory flag makes things run much faster
# compared to reading from the hdf5 file on disk, but requires significant
# memory (budget ~32GB conservatively)

if [ -z "$1" ]; then
    echo "usage: ./run_sw.sh EXP_NAME"
    exit 1
fi
i="$1"  # specify a name/tag of some sort to track this run

# ==== CONCEPT ====
# percent_novel = 1.0, reference_game = False, dataset = $DATASET
EXP_NAME="sw_concept_""$EX""_ex_$i"

python code/train.py --cuda --name "$EXP_NAME" --n_examples $EX --dataset $DATASET --percent_novel 1.0 --max_lang_length $MAXLEN --vocab_size $TOK --batch_size $BATCH_SIZE --accum_steps $ACCUM_STEPS --n_workers $N_WORKERS --backbone $BACKBONE --uniform_weight $UNIF --epochs $EPOCHS --load_shapeworld_into_memory

python code/sample.py "exp/$EXP_NAME"
python code/acre.py "exp/$EXP_NAME/sampled_lang.csv" --dataset $DATASET --cuda
python code/eval_zero_shot.py "exp/$EXP_NAME" --cuda

# Example of zero shot eval on other game types. If you need these to run faster specify less number of samples with the --n flag to sample.py
# python code/sample.py "exp/$EXP_NAME" --force_reference_game
# python code/acre.py "exp/$EXP_NAME/sampled_lang_force_ref.csv" --stats_only --dataset $REF_DATASET --cuda  # NOTE REF_DATASET SUPPLIED HERE!!

# python code/sample.py "exp/$EXP_NAME" --force_setref_game
# python code/acre.py "exp/$EXP_NAME/sampled_lang_force_setref.csv" --stats_only --dataset $DATASET --cuda

# ==== SETREF ====
# percent_novel = 0.0, reference_game = False, dataset = $DATASET
EXP_NAME="sw_setref_""$EX""_ex_$i"

python code/train.py --cuda --name "$EXP_NAME" --n_examples $EX --dataset $DATASET --percent_novel 0.0 --max_lang_length $MAXLEN --vocab_size $TOK --batch_size $BATCH_SIZE --accum_steps $ACCUM_STEPS --n_workers $N_WORKERS --backbone $BACKBONE --uniform_weight $UNIF --epochs $EPOCHS --load_shapeworld_into_memory

python code/sample.py $f
python code/acre.py "exp/$EXP_NAME/sampled_lang.csv" --dataset $DATASET --cuda
python code/eval_zero_shot.py "exp/$EXP_NAME" --cuda

# python code/sample.py "exp/$EXP_NAME" --force_reference_game
# python code/acre.py "exp/$EXP_NAME/sampled_lang_force_ref.csv" --stats_only --dataset $REF_DATASET --cuda  # NOTE REF_DATASET SUPPLIED HERE!!

# python code/sample.py "exp/$EXP_NAME" --force_concept_game
# python code/acre.py "exp/$EXP_NAME/sampled_lang_force_concept.csv" --stats_only --dataset $DATASET --cuda

# ==== REF ====
# percent_novel = 0.0, reference_game = True, dataset = $REF_DATASET (NOTE DIFFERENT DATASET!!)
EXP_NAME="sw_ref_""$EX""_ex_$i"

python code/train.py --cuda --name "$EXP_NAME" --n_examples $EX --dataset $REF_DATASET --percent_novel 0.0 --reference_game --max_lang_length $MAXLEN --vocab_size $TOK --batch_size $BATCH_SIZE --accum_steps $ACCUM_STEPS --n_workers $N_WORKERS --backbone $BACKBONE --uniform_weight $UNIF --epochs $EPOCHS --load_shapeworld_into_memory

python code/sample.py "exp/$EXP_NAME"
# Can't do ACRe for reference games - just track stats
python code/acre.py "exp/$EXP_NAME/sampled_lang.csv" --stats_only --dataset $REF_DATASET --cuda

# python code/sample.py "exp/$EXP_NAME" --force_setref_game
# python code/acre.py "exp/$EXP_NAME/sampled_lang_force_setref.csv" --stats_only --dataset $DATASET --cuda  # NOTE STANDARD DATASET SUPPLIED HERE!!

# python code/sample.py "exp/$EXP_NAME" --force_concept_game
# python code/acre.py "exp/$EXP_NAME/sampled_lang_force_concept.csv" --stats_only --dataset $DATASET --cuda

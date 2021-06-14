"""
io util
"""


from models.backbone import BACKBONES


def parse_args(defaults=False):
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Train", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", default="data/shapeworld/", type=str)
    parser.add_argument(
        "--test_dataset",
        default=None,
        type=str,
        help="Test dataset (if different) - will use test split",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        choices=list(BACKBONES.keys()),
        type=str,
        help="Vision backbone. If None, uses default in models.builder",
    )
    parser.add_argument(
        "--vocab_size",
        default=11,
        type=int,
        help="Communication vocab size (default is number of shapeworld shapes + colors)",
    )
    parser.add_argument(
        "--max_lang_length",
        default=4,
        type=int,
        help="Maximum language length, including SOS and EOS",
    )

    parser.add_argument(
        "--pretrained_feat_model",
        action="store_true",
        help="Use pretrained resnet18 as feature model for imgs",
    )
    arch_group_ = parser.add_argument_group(
        "architecture options", "mutually exclusive architecture options"
    )
    arch_group = arch_group_.add_mutually_exclusive_group()
    arch_group.add_argument(
        "--listener_only",
        action="store_true",
        help="Don't use teacher",
    )
    arch_group.add_argument(
        "--share_feat_model",
        action="store_true",
        help="Use same feature backbone for speaker and listener",
    )
    parser.add_argument(
        "--prototype",
        default="average",
        choices=["average", "transformer"],
        help="How to form prototypes",
    )
    parser.add_argument(
        "--n_transformer_heads",
        type=int,
        default=8,
        help="How many heads for multihead attention if --prototype transformer",
    )
    parser.add_argument(
        "--n_transformer_layers",
        type=int,
        default=2,
        help="How many transformer encoder layters if --prototype transformer",
    )
    parser.add_argument(
        "--joint_training",
        action="store_true",
        help="Jointly train teacher on classification task and communication task",
    )
    parser.add_argument(
        "--joint_training_lambda",
        type=float,
        default=1.0,
        help="Weight on joint training objective (comm task weight is 1.0)",
    )
    parser.add_argument(
        "--copy_listener",
        action="store_true",
        help="Pass teacher hidden state as communication channel",
    )
    parser.add_argument(
        "--reference_game",
        action="store_true",
        help="Use reference game - copy a single target over",
    )
    parser.add_argument(
        "--reference_game_xent",
        action="store_true",
        help="Use cross entropy loss for single reference game",
    )
    parser.add_argument(
        "--n_examples",
        default=None,
        type=int,
        help="# examples seen per agent (if none, automatically divide all images available); should be divisible by 2",
    )
    parser.add_argument(
        "--test_n_examples",
        default=None,
        type=int,
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--accum_steps",
        default=1,
        type=int,
        help="How often (in batches) to backprop - > 1 implies gradient accumulation",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="epsilon parameter for epsilon greedy search",
    )
    parser.add_argument(
        "--eps_anneal",
        type=float,
        default=0.0,
        help="Reduce --eps by this amount per epoch",
    )
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=1.0,
        help="softmax temp parameter for sampling (different from gumbel-softmax tau; note only works if > 1)",
    )
    parser.add_argument(
        "--softmax_temp_anneal",
        type=float,
        default=0.0,
        help="Reduce --softmax_temp by this amount per epoch",
    )
    parser.add_argument(
        "--uniform_weight",
        type=float,
        default=0.0,
        help="mix speaker outputs with this much of a uniform distribution",
    )
    parser.add_argument(
        "--uniform_weight_anneal",
        type=float,
        default=0.0,
        help="Reduce --uniform_weight by this amount per epoch",
    )
    parser.add_argument(
        "--no_cross_eval",
        action="store_true",
        help="don't evaluate across communication games",
    )
    parser.add_argument(
        "--percent_novel",
        default=1.0,
        type=float,
        help="pct chance that listener sees novel image",
    )
    parser.add_argument(
        "--test_percent_novel",
        default=None,
        type=float,
        help="test pct novel, if different from percent",
    )
    parser.add_argument(
        "--ignore_language",
        action="store_true",
        help="Ignore language - for MI random baseline",
    )
    parser.add_argument(
        "--ignore_examples",
        action="store_true",
        help="Ignore examples *after generation 0* only",
    )
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument(
        "--listener_reset_interval",
        default=0,
        type=int,
        help="How often in epochs to reset listener (if 0, no reset)",
    )
    parser.add_argument("--clip", default=100.0, type=float, help="Gradient clipping")
    parser.add_argument("--embedding_size", default=500, type=int)
    parser.add_argument("--speaker_hidden_size", default=1024, type=int)
    parser.add_argument("--listener_hidden_size", default=1024, type=int)
    parser.add_argument("--speaker_n_layers", default=2, type=int)
    parser.add_argument("--listener_n_layers", default=2, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--tau", default=1.0, type=float, help="Gumbel-softmax tau")
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument(
        "--save_interval",
        default=10,
        type=int,
        help="How often (in epochs) to save lang + model",
    )
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--transformer_lr", default=None, type=float)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="log with wandb")
    parser.add_argument("--wandb_project_name", default="cc", help="wandb project name")
    parser.add_argument("--n_workers", default=0, type=int)
    parser.add_argument("--name", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--vis", action="store_true", help="Visualize last batch of each epoch"
    )
    parser.add_argument(
        "--load_shapeworld_into_memory",
        action="store_true",
        help="Load shapeworld hdf5 into memory (can be high memory reqs)",
    )

    if defaults:
        args = parser.parse_args([])
    else:  # From CLI
        args = parser.parse_args()

    args.use_lang = not args.copy_listener and not args.listener_only

    if args.copy_listener and args.listener_only:
        parser.error(
            "argument --copy_listener: not allowed with argument --listener_only"
        )

    if args.reference_game_xent and not args.reference_game:
        parser.error("--reference_game_xent requires --reference_game")

    if args.test_percent_novel is None:
        args.test_percent_novel = args.percent_novel
    if args.test_n_examples is None:
        args.test_n_examples = args.n_examples

    if args.wandb:
        import wandb

        wandb.init(args.wandb_project_name, config=args)
        if args.name is not None:
            wandb.run.name = args.name
        else:
            args.name = wandb.run.name

    if args.name is None:
        args.name = "debug"

    return args

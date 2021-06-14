"""
Model building logic
"""


from . import base
from . import speaker
from . import listener

from .backbone import vision
from .backbone import feature
from .backbone import BACKBONES

from torch import optim, nn
import copy


from data.shapeworld import SHAPES, COLORS


DEFAULT_MODELS = {
    "shapeworld": vision.Conv4,
    "cub": vision.ResNet18,
}


def is_transformer_param(name):
    return name.startswith("speaker.transformer") or name.startswith("speaker.cls_emb")


def build_models(dataloaders, args):
    n_feats = dataloaders["train"].dataset.n_feats
    if len(n_feats) == 1:  # Feature based; use mlp

        def feat_fn(which):
            if which == "speaker":
                output_size = args.speaker_hidden_size
                n_layers = args.speaker_n_layers
            else:
                output_size = args.listener_hidden_size
                n_layers = args.listener_n_layers
            return feature.FeatureMLP(
                input_size=n_feats[0],
                output_size=output_size,
                n_layers=n_layers,
            )

    else:

        def feat_fn(which):
            # To use comm, make this conv4.
            if args.pretrained_feat_model:
                if dataloaders["train"].dataset.name == "shapeworld":
                    raise NotImplementedError
                return vision.PretrainedResNet18()
            else:
                if args.backbone is None:
                    return DEFAULT_MODELS[dataloaders["train"].dataset.name]()
                return BACKBONES[args.backbone]()

    if args.listener_only:
        speaker_feat_model = None
    else:
        speaker_feat_model = feat_fn("speaker")

    if args.share_feat_model:
        assert speaker_feat_model is not None
        listener_feat_model = speaker_feat_model
    else:
        listener_feat_model = feat_fn("listener")

    if args.listener_only or args.copy_listener:
        # Copy entire teacher internal state
        if args.listener_only:
            speaker_ = None
            speaker_size = None
        else:
            speaker_ = speaker.CopySpeaker(
                speaker_feat_model,
                dropout=args.dropout,
                prototype=args.prototype,
                n_transformer_heads=args.n_transformer_heads,
                n_transformer_layers=args.n_transformer_layers,
            )
            speaker_size = speaker_.emb_size
        listener_ = listener.CopyListener(
            listener_feat_model, message_size=speaker_size, dropout=args.dropout
        )
    else:
        # (account for SOS, EOS, UNK)
        speaker_embs = nn.Embedding(args.vocab_size + 3, args.embedding_size)
        listener_embs = nn.Embedding(args.vocab_size + 3, args.embedding_size)

        speaker_ = speaker.Speaker(
            speaker_feat_model,
            speaker_embs,
            hidden_size=args.speaker_hidden_size,
            dropout=args.dropout,
            tau=args.tau,
            prototype=args.prototype,
            n_transformer_heads=args.n_transformer_heads,
            n_transformer_layers=args.n_transformer_layers,
        )
        listener_ = listener.Listener(
            listener_feat_model,
            listener_embs,
            message_size=args.listener_hidden_size,
            dropout=args.dropout,
        )

    pair = base.Pair(speaker_, listener_)

    if args.cuda:
        pair = pair.cuda()

    # Optimization
    opt_params = [
        {
            "params": [
                p
                for name, p in pair.named_parameters()
                if not is_transformer_param(name)
            ],
            "lr": args.lr,
        }
    ]
    if args.prototype == "transformer":
        opt_params.append(
            {
                "params": [
                    p
                    for name, p in pair.named_parameters()
                    if is_transformer_param(name)
                ],
                "lr": args.transformer_lr
                if args.transformer_lr is not None
                else args.lr,
            }
        )

    optimizer = optim.Adam(opt_params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=10,
    )

    return {
        "pair": pair,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

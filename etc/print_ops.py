import sys

sys.path.append("./code")

from data.shapeworld import concept_to_lf, COLORS, SHAPES
import pandas as pd
from collections import defaultdict, Counter


def get_nots(concept_counts):
    not_records = []
    for feat in COLORS + SHAPES:
        neg_feat = f"not {feat}"

        feat_lf = concept_to_lf(feat)
        neg_feat_lf = concept_to_lf(neg_feat)

        feat_counts = concept_counts[feat_lf]
        neg_feat_counts = concept_counts[neg_feat_lf]

        feat_counts_norm = Counter(
            {k: v / sum(feat_counts.values()) for k, v in feat_counts.items()}
        )
        neg_feat_counts_norm = Counter(
            {k: v / sum(neg_feat_counts.values()) for k, v in neg_feat_counts.items()}
        )

        feat_lang, feat_pct = feat_counts_norm.most_common(1)[0]
        neg_feat_lang, neg_feat_pct = neg_feat_counts_norm.most_common(1)[0]

        not_records.append(
            (
                feat,
                feat_lang,
                feat_pct,
                neg_feat,
                neg_feat_lang,
                neg_feat_pct,
            )
        )
    not_df = pd.DataFrame.from_records(
        not_records,
        columns=["pos", "pos_lang", "pos_pct", "neg", "neg_lang", "neg_pct"],
    )
    return not_df


def get_binop(concept_counts, op="or"):
    or_records = []
    for feat1 in COLORS + SHAPES:
        for feat2 in COLORS + SHAPES:
            or_feat_a = f"{feat1} {op} {feat2}"
            or_feat_b = f"{feat1} {op} {feat2}"
            if concept_to_lf(or_feat_a) in concept_counts:
                or_feat = or_feat_a
            elif concept_to_lf(or_feat_b) in concept_counts:
                or_feat = or_feat_b
            else:
                continue

            feat1_lf = concept_to_lf(feat1)
            feat2_lf = concept_to_lf(feat2)
            or_feat_lf = concept_to_lf(or_feat)

            feat1_counts = concept_counts[feat1_lf]
            feat2_counts = concept_counts[feat2_lf]
            or_feat_counts = concept_counts[or_feat_lf]

            feat1_counts_norm = Counter(
                {k: v / sum(feat1_counts.values()) for k, v in feat1_counts.items()}
            )
            feat2_counts_norm = Counter(
                {k: v / sum(feat2_counts.values()) for k, v in feat2_counts.items()}
            )
            or_feat_counts_norm = Counter(
                {k: v / sum(or_feat_counts.values()) for k, v in or_feat_counts.items()}
            )

            feat1_lang, feat1_pct = feat1_counts_norm.most_common(1)[0]
            feat2_lang, feat2_pct = feat2_counts_norm.most_common(1)[0]
            or_feat_lang, or_feat_pct = or_feat_counts_norm.most_common(1)[0]

            or_records.append(
                (
                    feat1,
                    feat1_lang,
                    feat1_pct,
                    feat2,
                    feat2_lang,
                    feat2_pct,
                    or_feat,
                    or_feat_lang,
                    or_feat_pct,
                )
            )
    or_df = pd.DataFrame.from_records(
        or_records,
        columns=[
            "feat1",
            "feat1_lang",
            "feat1_pct",
            "feat2",
            "feat2_lang",
            "feat2_pct",
            f"{op}_feat",
            f"{op}_feat_lang",
            f"{op}_feat_pct",
        ],
    )
    return or_df


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="print ops", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("lang_file")

    args = parser.parse_args()

    lang = pd.read_csv(args.lang_file)

    concept_counts = defaultdict(Counter)

    for lang, concept in zip(lang["lang"], lang["true_lang"]):
        concept_lf = concept_to_lf(concept)
        # Remove SOS/EOS
        lang = lang[2:-2]
        concept_counts[concept_lf][lang] += 1

    not_df = get_nots(concept_counts)
    or_df = get_binop(concept_counts, "or")
    and_df = get_binop(concept_counts, "and")
    print(not_df)
    print(or_df)
    print(and_df)

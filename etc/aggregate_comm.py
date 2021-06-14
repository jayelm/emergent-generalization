# Check everything
import pandas as pd
import os
import json


CUB = {
    "ref": [
        ("./exp/0519channel_cub_ref_10_ex_1_3_3/", 1, 3, 3),
        ("./exp/0519channel_cub_ref_10_ex_1_5_5/", 1, 5, 5),
        ("./exp/0519channel_cub_ref_10_ex_1_20_100/", 1, 20, 100),
        ("./exp/0519channel_cub_ref_10_ex_1_20_1000/", 1, 20, 1000),
    ],
    "setref": [
        ("./exp/0519channel_cub_setref_10_ex_1_3_3/", 1, 3, 3),
        ("./exp/0519channel_cub_setref_10_ex_1_5_5/", 1, 5, 5),
        ("./exp/0519channel_cub_setref_10_ex_1_20_100/", 1, 20, 100),
        ("./exp/0519channel_cub_setref_10_ex_1_20_1000/", 1, 20, 1000),
    ],
    "concept": [
        ("./exp/0519channel_cub_concept_10_ex_1_3_3/", 1, 3, 3),
        ("./exp/0519channel_cub_concept_10_ex_1_5_5/", 1, 5, 5),
        ("./exp/0519channel_cub_concept_10_ex_1_20_100/", 1, 20, 100),
        ("./exp/0519channel_cub_concept_10_ex_1_20_1000/", 1, 20, 1000),
    ],
}


SW = {
    "ref": [
        ("./exp/0519channel_ref_20_ex_1_3_3/", 1, 3, 3),
        ("./exp/0519channel_ref_20_ex_1_5_5/", 1, 5, 5),
        ("./exp/0519channel_ref_20_ex_1_20_100/", 1, 20, 100),
        ("./exp/0519channel_ref_20_ex_1_20_1000/", 1, 20, 1000),
    ],
    "setref": [
        ("./exp/0519channel_setref_20_ex_1_3_3/", 1, 3, 3),
        ("./exp/0519channel_setref_20_ex_1_5_5/", 1, 5, 5),
        ("./exp/0519channel_setref_20_ex_1_20_100/", 1, 20, 100),
        ("./exp/0519channel_setref_20_ex_1_20_1000/", 1, 20, 1000),
    ],
    "concept": [
        ("./exp/0519channel_concept_20_ex_1_3_3/", 1, 3, 3),
        ("./exp/0519channel_concept_20_ex_1_5_5/", 1, 5, 5),
        ("./exp/0519channel_concept_20_ex_1_20_100/", 1, 20, 100),
        ("./exp/0519channel_concept_20_ex_1_20_1000/", 1, 20, 1000),
    ],
}


def load_all(runs, fname="sampled_lang_overall_stats.json", **kwargs):
    alls = []
    for (run, i, max_len, vocab_size) in runs:
        run = load_run(run, fname)
        if run:
            run.update(kwargs)
            run["length"] = max_len
            run["vocab_size"] = vocab_size
            alls.append(run)
    return alls


def load_run(run, fname):
    # Check that the run is done
    metrics_fname = os.path.join(run, "metrics.csv")
    if not os.path.exists(metrics_fname):
        print(f"Can't find {metrics_fname}, i.e. run doesn't exist")
        return {}
    ms = pd.read_csv(metrics_fname)
    assert 99 in ms["epoch"], f"Run {run} not done (max {max(ms['epoch'])})"

    # Check for sampled lang stats
    sampled_lang_stats_fname = os.path.join(run, fname)
    if os.path.exists(sampled_lang_stats_fname):
        with open(sampled_lang_stats_fname, "r") as f:
            return json.load(f)

    print(f"Can't find {sampled_lang_stats_fname}")
    return {}


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    args = parser.parse_args()

    fnames = [
        ("sampled_lang_overall_stats.json", "comm_entropy_results.csv"),
        ("sampled_stats.json", "comm_results.csv"),
    ]

    # Standard results - ref vs concept vs setref, standard eval
    for json_name, out_name in fnames:
        swc = load_all(
            SW["concept"], dataset="shapeworld", name="concept", fname=json_name
        )
        swsr = load_all(
            SW["setref"], dataset="shapeworld", name="setref", fname=json_name
        )
        swr = load_all(SW["ref"], dataset="shapeworld_ref", name="ref", fname=json_name)

        sw = swc + swsr + swr

        cc = load_all(CUB["concept"], dataset="cub", name="concept", fname=json_name)
        csr = load_all(CUB["setref"], dataset="cub", name="setref", fname=json_name)
        cr = load_all(CUB["ref"], dataset="cub", name="ref", fname=json_name)

        c = cc + csr + cr

        alls = sw + c
        pd.DataFrame.from_records(alls).to_csv(
            os.path.join("etc", out_name), index=False
        )

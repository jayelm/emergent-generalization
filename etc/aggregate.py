# Check everything
import pandas as pd
import os
import json


CUB = {
    "concept": [
        "./exp/28_concept_cub_10_ex_pt_1/",
        "./exp/28_concept_cub_10_ex_pt_2/",
        "./exp/28_concept_cub_10_ex_pt_3/",
        "./exp/28_concept_cub_10_ex_pt_4/",
        "./exp/28_concept_cub_10_ex_pt_5/",
    ],
    "setref": [
        "./exp/28_setref_cub_10_ex_pt_1/",
        "./exp/28_setref_cub_10_ex_pt_2/",
        "./exp/28_setref_cub_10_ex_pt_3/",
        "./exp/28_setref_cub_10_ex_pt_4/",
        "./exp/28_setref_cub_10_ex_pt_5/",
    ],
    "ref": [
        "./exp/28_ref_cub_10_ex_pt_1/",
        "./exp/28_ref_cub_10_ex_pt_2/",
        "./exp/28_ref_cub_10_ex_pt_3/",
        "./exp/28_ref_cub_10_ex_pt_4/",
        "./exp/28_ref_cub_10_ex_pt_5/",
    ],
    "ref_xent": [
        "./exp/311_ref_xent_cub_10_ex_pt_1/",
        "./exp/311_ref_xent_cub_10_ex_pt_2/",
        "./exp/311_ref_xent_cub_10_ex_pt_3/",
        "./exp/311_ref_xent_cub_10_ex_pt_4/",
        "./exp/311_ref_xent_cub_10_ex_pt_5/",
    ],
}


SW = {
    "concept": [
        "./exp/0501_concept_20_ex_1/",
        "./exp/0501_concept_20_ex_2/",
        "./exp/0501_concept_20_ex_3/",
        "./exp/0501_concept_20_ex_4/",
        "./exp/0501_concept_20_ex_5/",
    ],
    "setref": [
        "./exp/0501_setref_20_ex_1/",
        "./exp/0501_setref_20_ex_2/",
        "./exp/0501_setref_20_ex_3/",
        "./exp/0501_setref_20_ex_4/",
        "./exp/0501_setref_20_ex_5/",
    ],
    "ref": [
        "./exp/0501_ref_20_ex_1/",
        "./exp/0501_ref_20_ex_2/",
        "./exp/0501_ref_20_ex_3/",
        "./exp/0501_ref_20_ex_4/",
        "./exp/0501_ref_20_ex_5/",
    ],
    "ref_xent": [
        "./exp/0501_ref_xent_20_ex_1/",
        "./exp/0501_ref_xent_20_ex_2/",
        "./exp/0501_ref_xent_20_ex_3/",
        "./exp/0501_ref_xent_20_ex_4/",
        "./exp/0501_ref_xent_20_ex_5/",
    ],
}


def load_all(runs, fname="sampled_lang_overall_stats.json", **kwargs):
    alls = []
    for i, run in enumerate(runs):
        run = load_run(run, fname)
        if run:
            run.update(kwargs)
            run["n"] = i
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
        ("sampled_lang_overall_stats.json", "entropy_results.csv"),
        ("sampled_lang_force_ref_overall_stats.json", "entropy_force_ref_results.csv"),
        ("sampled_stats_force_ref.json", "force_ref_results.csv"),
        (
            "sampled_lang_force_concept_overall_stats.json",
            "entropy_force_concept_results.csv",
        ),
        ("sampled_stats_force_concept.json", "force_concept_results.csv"),
        (
            "sampled_lang_force_setref_overall_stats.json",
            "entropy_force_setref_results.csv",
        ),
        ("sampled_stats_force_setref.json", "force_setref_results.csv"),
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

    # Ref xent results - all standard eval (i.e. no need to force)
    fnames = [
        ("sampled_lang_overall_stats.json", "xent_entropy_results.csv"),
        ("sampled_stats.json", "xent_results.csv"),
    ]
    for json_name, out_name in fnames:
        swc = load_all(
            SW["concept"], dataset="shapeworld", name="concept", fname=json_name
        )
        swsr = load_all(
            SW["setref"], dataset="shapeworld", name="setref", fname=json_name
        )
        swr = load_all(SW["ref"], dataset="shapeworld_ref", name="ref", fname=json_name)
        swrx = load_all(
            SW["ref_xent"], dataset="shapeworld_ref", name="ref_xent", fname=json_name
        )

        sw = swc + swsr + swr + swrx

        cc = load_all(CUB["concept"], dataset="cub", name="concept", fname=json_name)
        csr = load_all(CUB["setref"], dataset="cub", name="setref", fname=json_name)
        cr = load_all(CUB["ref"], dataset="cub", name="ref", fname=json_name)
        crx = load_all(CUB["ref_xent"], dataset="cub", name="ref_xent", fname=json_name)

        c = cc + csr + cr + crx

        alls = sw + c
        pd.DataFrame.from_records(alls).to_csv(
            os.path.join("etc", out_name), index=False
        )

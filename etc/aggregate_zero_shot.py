# Check everything
import pandas as pd
import os
import json


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
}


def load_all(runs, **kwargs):
    alls = []
    for i, run in enumerate(runs):
        run_records = load_run(run, **kwargs, i=i)
        alls.extend(run_records)
    return alls


def load_run(run, **kwargs):
    # Check that the run is done
    run_records = []
    for split in ["train", "test"]:
        eval_fname = os.path.join(run, f"zero_shot_{split}.json")
        if not os.path.exists(eval_fname):
            print(f"Can't find {eval_fname}")
            continue

        with open(eval_fname, "r") as f:
            eval_json = json.load(f)
        for metric, value in eval_json.items():
            *lang_type, metric = metric.split("_")
            lang_type = "_".join(lang_type)
            record = {
                "metric": metric,
                "lang_type": lang_type,
                "value": value,
                "split": split,
            }
            record.update(**kwargs)
            run_records.append(record)
    return run_records


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    args = parser.parse_args()

    swc = load_all(SW["concept"], dataset="cub", name="concept")
    swsr = load_all(SW["setref"], dataset="cub", name="setref")

    pd.DataFrame.from_records(swc + swsr).to_csv(
        "etc/zero_shot_results.csv", index=False
    )

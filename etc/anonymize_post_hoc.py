import sys

sys.path.append("./code")
from acre import anonymize

import pandas as pd
import os
import argparse


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("files", nargs="+")

    args = parser.parse_args()

    for file in args.files:
        df = pd.read_csv(file)
        langs = df["lang"].str.split(" ")
        if not langs[0][0].isnumeric():
            print(
                f"{file} already seems anonymized (first string: {df['lang'][0]}), skipping"
            )
            continue
        langs = [anonymize(lang) for lang in langs]
        langs = ["".join(lang) for lang in langs]
        df["lang"] = langs
        print(file)
        df.to_csv(file, index=False)

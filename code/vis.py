"""
Visualize communication
"""


import os
from jinja2 import Template
from PIL import Image
import numpy as np


def sanitize(x):
    return x.replace("<s>", " ").replace("</s>", ".")


def report(
    spk_inp,
    spk_y,
    lis_inp,
    lis_y,
    dataset,
    epoch,
    split,
    lang_texts,  # lang_type -> [batch_size, lang_text]
    true_lang_text,
    lis_preds,  # lang_type -> [batch_size, n_preds]
    exp_dir="vis",
):
    exp_dir = os.path.join(exp_dir, "html")
    img_dir = os.path.join(exp_dir, "images")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    games = []
    for game_i, true_lang in enumerate(true_lang_text):
        spk_info = []
        lis_infos = {lang_type: [] for lang_type in lang_texts.keys()}

        lang_text_sanitized = {
            lang_type: sanitize(lang[game_i]) for lang_type, lang in lang_texts.items()
        }
        lis_accs = {
            lang_type: (
                lis_y[game_i].cpu().numpy() == lis_pred[game_i].cpu().numpy()
            ).mean()
            for lang_type, lis_pred in lis_preds.items()
        }

        infos = [
            ("speaker", 0, "true", spk_inp, spk_y, spk_y, spk_info),
        ]
        for lang_type_i, lang_type in enumerate(lang_texts.keys()):
            li = lis_infos[lang_type]
            lp = lis_preds[lang_type]
            infos.append(("listener", lang_type_i, lang_type, lis_inp, lis_y, lp, li))

        for model, lang_type_i, lang_type, inps, labels, preds, info in infos:
            for i in range(inps.shape[1]):

                t = int(labels[game_i, i])
                p = int(preds[game_i, i])

                # Generate the html code (and possibly do side effects e.g.
                # save image) for this visualization
                inp_vis = dataset.vis_input(
                    inps[game_i, i],
                    name=model,
                    epoch=epoch,
                    split=split,
                    game_i=game_i,
                    i=i,
                    exp_dir=exp_dir,
                    # Overwrite if this is the first lang type
                    overwrite=lang_type_i == 0,
                )
                info.append(
                    {
                        "gt": t,
                        "pred": p,
                        "correct": "correct" if t == p else "incorrect",
                        "visualization": inp_vis,
                    }
                )

        games.append(
            {
                "i": game_i,
                "true_text": sanitize(true_lang),
                "pred_text": lang_text_sanitized,
                "speaker_info": spk_info,
                "listener_info": lis_infos,
                "accuracy": lis_accs,
            }
        )

    report_fname = os.path.join(exp_dir, f"{epoch}_{split}.html")

    with open("html/vis.j2", "r") as f:
        report_template = Template(f.read())

    report_html = report_template.render(games=games)
    with open(report_fname, "w") as f:
        f.write(report_html)

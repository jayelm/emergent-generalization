PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def init_vocab(langs):
    i2w = {
        PAD_IDX: PAD_TOKEN,
        SOS_IDX: SOS_TOKEN,
        EOS_IDX: EOS_TOKEN,
        UNK_IDX: UNK_TOKEN,
    }
    w2i = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }
    for lang in langs:
        for tok in lang:
            if tok not in w2i:
                i = len(w2i)
                w2i[tok] = i
                i2w[i] = tok
    return {
        "w2i": w2i,
        "i2w": i2w,
        "size": len(w2i),
        "pad_idx": PAD_IDX,
        "unk_idx": UNK_IDX,
        "eos_idx": EOS_IDX,
        "sos_idx": SOS_IDX,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "eos_token": EOS_TOKEN,
        "sos_token": SOS_TOKEN,
    }

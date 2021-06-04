"""
Combine speaker and listener for easier training
"""

from torch import nn


class Pair(nn.Module):
    def __init__(self, speaker, listener):
        super().__init__()
        self.speaker = speaker
        self.listener = listener
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.xent_criterion = nn.CrossEntropyLoss()

"""
RNNs/language models etc
"""


import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn


class RNNEncoder(nn.Module):
    """
    RNN Encoder - takes in onehot representations of tokens, rather than numeric
    """

    def __init__(self, embedding_module, hidden_size=100):
        super().__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, hidden_size)

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = seq @ self.embedding.weight

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist(),
        )

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.gru.reset_parameters()

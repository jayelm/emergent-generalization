"""
sequence-to-sequence models for acre/compositionality
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from data import language


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.rnn = nn.GRU(self.embed_dim, self.hidden_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

    def forward(self, caps, caplens):
        emb = self.embedding(caps)

        packed_input = pack_padded_sequence(
            emb,
            caplens.cpu(),
            enforce_sorted=False,
            batch_first=True,
        )
        _, hidden = self.rnn(packed_input)
        return hidden[-1]

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.embedding.reset_parameters()


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, vocab_size, encoder_dim, embed_dim, decoder_dim, dropout=0.0):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)

        self.fc = nn.Linear(
            decoder_dim, self.vocab_size
        )  # linear layer to find scores over vocabulary

        self.init_weights()  # Initialize some layers with the uniform distribution
        self.decoder = nn.GRU(embed_dim, decoder_dim, batch_first=True)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, enc_out, out_seq, out_len):
        """
        Forward propagation.

        :param encoder_out:
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices

        Note this method takes standard language input (i.e. each value is in [0, N) where N is size of vocab)
        Compare to sampling method which returns one-hot output (for gumbel softmax trick)
        """
        # Embedding
        out_emb = self.embedding(out_seq)

        # Initialize LSTM state
        h = enc_out.unsqueeze(0)

        # We won't decode at the <end> position, since we've finished
        # generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_len = (out_len - 1).cpu()

        inp = pack_padded_sequence(
            out_emb,
            decode_len,
            enforce_sorted=False,
            batch_first=True,
        )
        targets = pack_padded_sequence(
            out_seq[:, 1:],
            decode_len,
            enforce_sorted=False,
            batch_first=True,
        ).data

        output, _ = self.decoder(inp, h)
        predictions = self.fc(self.dropout(output.data))

        return predictions, targets

    def sample(
        self,
        encoder_out,
        greedy=False,
        max_length=50,
        trim=True,
        return_all_scores=False,
    ):
        """
        Sample from the decoder.

        :param trim: if True, trim captions to less than max_length based on max
        cap length sampled
        """
        with torch.no_grad():
            batch_size = encoder_out.size(0)
            h = encoder_out.unsqueeze(0)

            # Contains series of sampled onehot vectors
            lang_tensor = torch.full(
                (batch_size, max_length), language.PAD_IDX, dtype=torch.int64
            ).to(encoder_out.device)
            # And vector lengths
            lang_length = torch.ones(batch_size, dtype=torch.int64).to(
                encoder_out.device
            )
            scores = torch.zeros(batch_size, dtype=torch.float32).to(encoder_out.device)
            done_sampling = torch.zeros(batch_size, dtype=torch.uint8).to(
                encoder_out.device
            )
            if return_all_scores:
                # No scores for the first token
                all_scores = torch.full(
                    (batch_size, max_length - 1, self.vocab_size),
                    -np.inf,
                    dtype=torch.float32,
                )

            # first input is SOS token
            # (batch_size, n_vocab)
            inputs = torch.full(
                (batch_size, 1), language.SOS_IDX, dtype=torch.int64
            ).to(encoder_out.device)

            # Add SOS to lang
            lang_tensor[:, 0] = inputs.squeeze(1)

            # compute embeddings
            inputs = self.embedding(inputs)

            for i in range(1, max_length - 1):
                if done_sampling.all():
                    break

                outputs, h = self.decoder(inputs, h)  # (batch_size,
                outputs = outputs.squeeze(1)  # (batch_size, hidden_size)
                outputs = self.fc(self.dropout(outputs))  # (batch_size, vocab_size)
                if greedy:
                    outputs = torch.log_softmax(outputs, dim=1)
                    predicted = outputs.argmax(1, keepdim=True)
                    predicted_score = torch.gather(outputs, 1, predicted)
                    predicted = predicted.squeeze(1)
                    predicted_score = predicted_score.squeeze(1)
                    if return_all_scores:
                        all_scores[:, i - 1] = outputs
                else:
                    dist = Categorical(logits=outputs)
                    predicted = dist.sample()
                    predicted_score = dist.log_prob(predicted)
                    if return_all_scores:
                        all_scores[:, i - 1] = dist.logits

                lang_tensor[:, i] = predicted

                # Add 1 where not done smapling
                not_done_sampling = 1 - done_sampling
                lang_length += not_done_sampling

                # Add probability where not done sampling
                scores += predicted_score * not_done_sampling

                # Update done sampling where predicted is end idx (and not already done sampling)
                sampled_end = (lang_tensor[:, i] == language.EOS_IDX).byte()
                done_sampling = done_sampling | sampled_end

                inputs = self.embedding(predicted.unsqueeze(1))

            # Add EOS as last token if we never sampled it
            lang_tensor[:, -1] = language.EOS_IDX

            # Final add 1 where not done samping (EOS)
            lang_length += 1 - done_sampling

            # Trim max length
            if trim:
                max_lang_len = lang_length.max()
                lang_tensor = lang_tensor[:, :max_lang_len]

            if return_all_scores:
                return lang_tensor, lang_length, scores, all_scores
            return lang_tensor, lang_length, scores

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.fc.reset_parameters()

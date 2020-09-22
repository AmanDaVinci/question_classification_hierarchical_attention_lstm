import torch
import torch.nn as nn
import numpy as np

from question_classification.models.highway import Highway


class HierRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, low_hidden_size, high_hidden_size, num_classes, max_character_len, max_sentence_len, use_highway=False):
        # # TODO: Workout the dimensions.
        # TODO: Make it work with att lstm
        super(HierRNN, self).__init__()
        # The five layers of the hierchical lstm:
        self.character_embedding = nn.Embedding(num_embeddings, embedding_dim)

        if use_highway:
            self.highway = Highway(low_hidden_size * max_character_len, low_hidden_size * max_character_len)
        self.low_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=low_hidden_size, batch_first=True)
        self.high_lstm = nn.LSTM(input_size=low_hidden_size * max_character_len, hidden_size=high_hidden_size, batch_first=True)
        self.output_layer = nn.Linear(high_hidden_size * max_sentence_len, num_classes)
        self.low_hidden_size = low_hidden_size

        self.use_highway = use_highway

    def low_reset_states(self, batch_size):
        return torch.zeros(batch_size, self.low_hidden_size), torch.zeros(batch_size, self.low_hidden_size)

    def high_reset_states(self, batch_size):
        return torch.zeros(batch_size, self.high_hidden_size), torch.zeros(batch_size, self.high_hidden_size)

    def forward(self, x):
        # # TODO: forward logic, does it work like this?
        sentence = []
        batch_size = x.size(0)
        for i in range(x.size(1)):
            embedded_word = self.character_embedding(x[:, i, :])
            hidden = self.low_reset_states(batch_size)
            low_out, hidden = self.low_lstm(embedded_word)
            word_repr = low_out.contiguous().view((batch_size, - 1))
            if self.use_highway:
                word_repr = self.highway(word_repr)
            sentence.append(word_repr)
        sentence = torch.stack(sentence)
        sentence_repr, hidden = self.high_lstm(sentence)
        output = self.output_layer(sentence_repr.contiguous().view((batch_size, - 1)))
        return output

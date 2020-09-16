import torch
import torch.nn as nn
import numpy as np

from highway import Highway


class HierRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, low_hidden_size, high_hidden_size, num_classes, max_character_len, max_sentence_len):
        # # TODO: Workout the dimensions.
        # TODO: Make it work with att lstm

        # The five layers of the hierchical lstm:
        self.character_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.highway = Highway(low_hidden_size * max_character_len, low_hidden_size)
        # self.low_attlstm = something
        # self.high_attlstm = something
        self.low_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=low_hidden_size)
        self.high_lstm = nn.LSTM(input_size=low_hidden_size, hidden_size=high_hidden_size)
        self.output_layer = nn.Linear(high_hidden_size * max_sentence_len, num_classes)
        self.low_hidden_size = low_hidden_size

    def low_reset_states(self, batch_size):
        return torch.zeros(batch_size, self.low_hidden_size), torch.zeros(batch_size, self.low_hidden_size)

    def high_reset_states(self, batch_size):
        return torch.zeros(batch_size, self.high_hidden_size), torch.zeros(batch_size, self.high_hidden_size)

    def forward(self, input):
        # # TODO: forward logic, does it work like this?
        sentence = []
        batch_size = input.size(1)
        for word in input:
            embedded_word = self.character_embedding(word)
            hidden = self.low_reset_states(batch_size)
            low_out, hidden = self.low_lstm(embedded_word, hidden)
            word_repr = self.highway(low_out.flatten())
            sentence.append(word_repr)
        sentence = torch.stack(sentence)
        hidden = self.high_reset_states(batch_size)
        sentence_repr, hidden = self.high_lstm(sentence, hidden)
        output = self.output_layer(sentence_repr.flatten())
        return output

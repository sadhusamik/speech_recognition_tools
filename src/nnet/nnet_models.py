import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class nnetFeedforward(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size, out_size):
        super(nnetFeedforward, self).__init__()
        input_sizes = [input_size] + [hidden_size] * num_layers
        output_sizes = [hidden_size] * num_layers + [out_size]

        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for (in_size, out_size) in zip(input_sizes, output_sizes)])

        self.relu = nn.ReLU()

    def forward(self, inputs):
        embeds = []
        for layer in self.layers[0:-1]:
            inputs = self.relu(layer(inputs))
            embeds.append(inputs)

        return embeds, self.layers[-1](inputs)

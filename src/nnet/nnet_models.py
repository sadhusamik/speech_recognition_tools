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
            inputs = layer(inputs)
            embeds.append(inputs)  # Tap network before activation
            inputs = self.relu(inputs)

        return embeds, self.layers[-1](inputs)


class encoderRNN(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size, bn_size):
        super(encoderRNN, self).__init__()

        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.bottleneck = nn.Conv1d(in_channels=hidden_size, out_channels=bn_size, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        packed_rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):
            packed_rnn_inputs, _ = layer(packed_rnn_inputs)

        inputs, _ = pad_packed_sequence(packed_rnn_inputs, True, total_length=seq_len)

        inputs = self.relu(self.bottleneck(torch.transpose(inputs, 1, 2)))

        return torch.transpose(inputs, 1, 2)


class decoderRNN(nn.Module):

    def __init__(self, bn_size, num_layers, hidden_size, out_size):
        super(decoderRNN, self).__init__()
        input_sizes = [bn_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.regression = nn.Conv1d(in_channels=hidden_size, out_channels=out_size, kernel_size=1, stride=1)

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        packed_rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):
            packed_rnn_inputs, _ = layer(packed_rnn_inputs)

        inputs, _ = pad_packed_sequence(packed_rnn_inputs, True, total_length=seq_len)
        inputs = self.regression(torch.transpose(inputs, 1, 2))

        return torch.transpose(inputs, 1, 2)


class nnetAEClassifierMultitask(nn.Module):

    def __init__(self, input_size, out_size, num_layers_enc, num_layers_class, num_layers_ae, hidden_size, bn_size):
        super(nnetAEClassifierMultitask, self).__init__()

        self.encoder = encoderRNN(input_size, num_layers_enc, hidden_size, bn_size)
        self.classifier = decoderRNN(bn_size, num_layers_class, hidden_size, out_size)
        self.ae = decoderRNN(bn_size, num_layers_ae, hidden_size, input_size)

    def forward(self, inputs, lengths):
        return self.classifier(self.encoder(inputs, lengths), lengths), self.ae(self.encoder(inputs, lengths), lengths)


class nnetAEClassifierMultitaskAEAR(nn.Module):

    def __init__(self, input_size, out_size, num_layers_enc, num_layers_class, num_layers_ae, hidden_size, bn_size,
                 time_shift):
        super(nnetAEClassifierMultitaskAEAR, self).__init__()

        self.encoder = encoderRNN(input_size, num_layers_enc, hidden_size, bn_size)
        self.classifier = decoderRNN(bn_size, num_layers_class, hidden_size, out_size)
        self.ae = decoderRNN(bn_size, num_layers_ae, hidden_size, input_size)
        self.ar = decoderRNN(bn_size, num_layers_ae, hidden_size, input_size)
        self.time_shift = time_shift

    def forward(self, inputs, lengths):
        return self.classifier(self.encoder(inputs, lengths), lengths), \
               self.ae(self.encoder(inputs, lengths), lengths), \
               self.ar(self.encoder(inputs[:, :-self.time_shift, :], lengths - self.time_shift),
                       lengths - self.time_shift)

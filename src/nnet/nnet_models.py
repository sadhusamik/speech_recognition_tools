import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class nnetFeedforward(nn.Module):
    """
    Simple linear feedfoward classification network
    """

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


class nnetLinearWithConv(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size, out_size):
        super(nnetLinearWithConv, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * (num_layers - 1) + [out_size]
        self.layers = nn.ModuleList(
            [nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])
        self.relu = nn.ReLU()

    def forward(self, inputs, lengths):
        inputs = torch.transpose(inputs, 1, 2)
        for i, layer in enumerate(self.layers[:-1]):
            inputs = self.relu(layer(inputs))
        inputs = self.layers[-1](inputs)

        return torch.transpose(inputs, 1, 2)


class nnetRNN(nn.Module):
    """
    RNN classification network
    """

    def __init__(self, input_size, num_layers, hidden_size, out_size, dropout):
        super(nnetRNN, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])
        self.dropout = nn.Dropout(dropout)
        self.regression = nn.Conv1d(in_channels=hidden_size, out_channels=out_size, kernel_size=1, stride=1)

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):
            rnn_inputs, _ = layer(rnn_inputs)

            rnn_inputs, _ = pad_packed_sequence(
                rnn_inputs, True, total_length=seq_len)

            if i + 1 < len(self.layers):
                rnn_inputs = self.dropout(rnn_inputs)

            rnn_inputs = pack_padded_sequence(rnn_inputs, lengths, True)

        rnn_inputs, _ = pad_packed_sequence(rnn_inputs, True, total_length=seq_len)

        rnn_inputs = self.regression(torch.transpose(rnn_inputs, 1, 2))

        return torch.transpose(rnn_inputs, 1, 2)


class rnnSubnet(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size):
        super(rnnSubnet, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):
            layer.flatten_parameters()
            rnn_inputs, _ = layer(rnn_inputs)

            rnn_inputs, _ = pad_packed_sequence(
                rnn_inputs, True, total_length=seq_len)

            rnn_inputs = pack_padded_sequence(rnn_inputs, lengths, True)

        rnn_inputs, _ = pad_packed_sequence(rnn_inputs, True, total_length=seq_len)

        return rnn_inputs


class nnetRNNMultimod(nn.Module):

    def __init__(self, input_size, num_layers_subband, num_layers, hidden_size_subband, out_size, mod_num):
        super(nnetRNNMultimod, self).__init__()

        self.mod_num = mod_num
        subnets = []
        for i in range(mod_num):
            subnets.append(rnnSubnet(input_size, num_layers_subband, hidden_size_subband))

        self.subnets = nn.ModuleList(subnets)

        input_sizes = [mod_num * hidden_size_subband] * num_layers
        output_sizes = [mod_num * hidden_size_subband] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])
        self.regression = nn.Conv1d(in_channels=mod_num * hidden_size_subband, out_channels=out_size, kernel_size=1,
                                    stride=1)

    def forward(self, inputs, lengths):
        seq_len = inputs[0].size(1)
        sub_out = []
        for sn in range(self.mod_num):
            sub_out.append(self.subnets[sn](inputs[sn], lengths))
        inputs = torch.cat(sub_out, dim=2)

        rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):
            rnn_inputs, _ = layer(rnn_inputs)

            rnn_inputs, _ = pad_packed_sequence(
                rnn_inputs, True, total_length=seq_len)

            rnn_inputs = pack_padded_sequence(rnn_inputs, lengths, True)
        rnn_inputs, _ = pad_packed_sequence(rnn_inputs, True, total_length=seq_len)
        rnn_inputs = self.regression(torch.transpose(rnn_inputs, 1, 2))

        return torch.transpose(rnn_inputs, 1, 2)


class encoderRNN(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size, bn_size, dropout):
        super(encoderRNN, self).__init__()

        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.bottleneck = nn.Conv1d(in_channels=hidden_size, out_channels=bn_size, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):

            rnn_inputs, _ = layer(rnn_inputs)

            rnn_inputs, _ = pad_packed_sequence(
                rnn_inputs, True, total_length=seq_len)

            if i + 1 < len(self.layers):
                rnn_inputs = self.dropout(rnn_inputs)

            rnn_inputs = pack_padded_sequence(rnn_inputs, lengths, True)

        inputs, _ = pad_packed_sequence(rnn_inputs, True, total_length=seq_len)

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

    def __init__(self, input_size, out_size, num_layers_enc, num_layers_class, num_layers_ae, hidden_size, bn_size,
                 dropout):
        super(nnetAEClassifierMultitask, self).__init__()

        self.encoder = encoderRNN(input_size, num_layers_enc, hidden_size, bn_size, dropout)
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


class VAEEncoder(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, bn_size, dropout):
        super(VAEEncoder, self).__init__()

        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.dropout = nn.Dropout(dropout)
        self.means = nn.Conv1d(in_channels=hidden_size, out_channels=bn_size, kernel_size=1, stride=1)
        self.vars = nn.Conv1d(in_channels=hidden_size, out_channels=bn_size, kernel_size=1, stride=1)

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):

            rnn_inputs, _ = layer(rnn_inputs)

            rnn_inputs, _ = pad_packed_sequence(
                rnn_inputs, True, total_length=seq_len)

            if i + 1 < len(self.layers):
                rnn_inputs = self.dropout(rnn_inputs)

            rnn_inputs = pack_padded_sequence(rnn_inputs, lengths, True)

        inputs, _ = pad_packed_sequence(rnn_inputs, True, total_length=seq_len)

        means = self.means(torch.transpose(inputs, 1, 2))
        vars = self.vars(torch.transpose(inputs, 1, 2))
        return torch.transpose(means, 1, 2), torch.transpose(vars, 1, 2), inputs


class VAEEncoderTransformer(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, bn_size, dropout, nhead=16):
        super(VAEEncoderTransformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size,
                                                   dropout=dropout, batch_first=True)
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.means = nn.Conv1d(in_channels=hidden_size, out_channels=bn_size, kernel_size=1, stride=1)
        self.vars = nn.Conv1d(in_channels=hidden_size, out_channels=bn_size, kernel_size=1, stride=1)

    def forward(self, inputs, lengths):
        inputs = self.layers(inputs)
        means = self.means(torch.transpose(inputs, 1, 2))
        vars = self.vars(torch.transpose(inputs, 1, 2))

        return torch.transpose(means, 1, 2), torch.transpose(vars, 1, 2), inputs


class VAEDecoderTransformer(nn.Module):
    def __init__(self, bn_size, num_layers, hidden_size, input_size, dropout, nhead=16):
        super(VAEDecoderTransformer, self).__init__()

        decoder_layer = nn.TransformerEncoderLayer(d_model=bn_size, nhead=nhead, dim_feedforward=hidden_size,
                                                   dropout=dropout, batch_first=True)

        self.layers = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.means = nn.Conv1d(in_channels=hidden_size, out_channels=input_size, kernel_size=1, stride=1)
        self.vars = nn.Conv1d(in_channels=hidden_size, out_channels=input_size, kernel_size=1, stride=1)

    def forward(self, inputs, lengths):
        inputs = self.layers(inputs)
        means = self.means(torch.transpose(inputs, 1, 2))

        return torch.transpose(means, 1, 2)


class VAEDecoder(nn.Module):
    def __init__(self, bn_size, num_layers, hidden_size, input_size):
        super(VAEDecoder, self).__init__()

        input_sizes = [bn_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.means = nn.Conv1d(in_channels=hidden_size, out_channels=input_size, kernel_size=1, stride=1)
        self.vars = nn.Conv1d(in_channels=hidden_size, out_channels=input_size, kernel_size=1, stride=1)

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        packed_rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):
            packed_rnn_inputs, _ = layer(packed_rnn_inputs)

        inputs, _ = pad_packed_sequence(packed_rnn_inputs, True, total_length=seq_len)

        means = self.means(torch.transpose(inputs, 1, 2))

        return torch.transpose(means, 1, 2)


class latentSampler(nn.Module):
    def __init__(self, use_gpu=True):
        super(latentSampler, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, latent):
        if self.use_gpu:
            return latent[0] + torch.exp(latent[1]) * torch.randn(latent[0].shape).cuda()
        else:

            return latent[0] + torch.exp(latent[1]) * torch.randn(latent[0].shape)


class nnetVAEClassifier(nn.Module):
    def __init__(self, input_size, out_size, num_layers_enc, num_layers_class, num_layers_ae, hidden_size, bn_size,
                 dropout, use_gpu=True):
        super(nnetVAEClassifier, self).__init__()

        self.vae_encoder = VAEEncoder(input_size, num_layers_enc, hidden_size, bn_size, dropout)
        self.vae_decoder = VAEDecoder(bn_size, num_layers_ae, hidden_size, input_size)
        self.sampler = latentSampler(use_gpu)
        self.classifier = decoderRNN(bn_size, num_layers_class, hidden_size, out_size)

    def forward(self, inputs, lengths):
        latent = self.vae_encoder(inputs, lengths)
        inputs = self.sampler(latent)
        return self.classifier(inputs, lengths), self.vae_decoder(inputs, lengths), latent


class nnetVAE(nn.Module):
    """
        A Variational Autoencoder (VAE) implementation in pyTorch
    """

    def __init__(self, input_size, num_layers_enc, num_layers_dec, hidden_size, bn_size,
                 dropout, use_gpu=True, only_AE=False, use_transformer=False):
        super(nnetVAE, self).__init__()

        self.bn_size = bn_size
        self.gpu = use_gpu
        if use_transformer:
            self.vae_encoder=VAEEncoderTransformer(input_size, num_layers_enc, hidden_size, bn_size, dropout)
        else:
            self.vae_encoder = VAEEncoder(input_size, num_layers_enc, hidden_size, bn_size, dropout)
        if use_transformer:
            self.vae_decoder = VAEDecoderTransformer(bn_size, num_layers_dec, hidden_size, input_size)
        else:
            self.vae_decoder = VAEDecoder(bn_size, num_layers_dec, hidden_size, input_size)

        self.sampler = latentSampler(use_gpu)
        self.only_AE = only_AE

    def forward(self, inputs, lengths):
        latent = self.vae_encoder(inputs, lengths)
        if self.only_AE:
            return self.vae_decoder(latent[0], lengths), latent
        else:
            inputs = self.sampler(latent)
            return self.vae_decoder(inputs, lengths), latent

    def vae_loss(self, x, ae_out, latent_out, out_dist='gauss'):
        import sys
        if out_dist == 'gauss':
            log_lhood = torch.mean(-0.5 * torch.pow((x - ae_out), 2) - 0.5 * np.log(2 * np.pi * 1))
        elif out_dist == 'laplace':
            log_lhood = torch.mean(-torch.abs(x - ae_out) - np.log(2))
        else:
            print("Output distribution of VAE can be 'gauss' or 'laplace'")
            sys.exit(1)

        kl_loss = 0.5 * torch.mean(
            1 - torch.pow(latent_out[0], 2) - torch.pow(torch.exp(latent_out[1]), 2) + 2 * latent_out[1])
        return log_lhood, kl_loss

    def compute_llhood(self, inputs, lengths, sample_num=10, out_dist='gauss'):
        latent = self.vae_encoder(inputs, lengths)
        latent_out = (latent[0][0], latent[1][0])

        loss_acc_recon = 0
        loss_acc_kl = 0
        for i in range(sample_num):
            z = self.sampler(latent)
            loss = self.vae_loss(inputs[0], self.vae_decoder(z, lengths)[0], latent_out, out_dist)
            loss_acc_recon += loss[0].item()
            loss_acc_kl -= loss[1].item()

        return loss_acc_recon / sample_num, loss_acc_kl / sample_num

    def generate(self, size=512):

        if self.gpu:
            input = torch.randn([1, size, self.bn_size]).cuda()
        else:
            input = torch.randn([1, size, self.bn_size])

        return self.vae_decoder(input, torch.IntTensor([size]))


class nnetARVAE(nn.Module):
    def __init__(self, input_size, num_layers_enc, num_layers_dec, hidden_size, bn_size,
                 dropout, num_outs, use_gpu=True):
        super(nnetARVAE, self).__init__()
        self.num_outs = num_outs
        self.vae_encoder = VAEEncoder(input_size, num_layers_enc, hidden_size, bn_size, dropout)
        decoders = []
        for i in range(num_outs):
            decoders.append(VAEDecoder(bn_size, num_layers_dec, hidden_size, input_size))
        self.vae_decoder = torch.nn.ModuleList(decoders)
        self.sampler = latentSampler(use_gpu)

    def forward(self, inputs, lengths):
        latent = self.vae_encoder(inputs, lengths)
        inputs = self.sampler(latent)
        return torch.cat([self.vae_decoder[i](inputs, lengths)[None, :] for i in range(self.num_outs)]), latent


class VAEEncodedClassifier(nn.Module):

    def __init__(self, vae_model, input_size, num_layers, hidden_size, out_size):
        super(VAEEncodedClassifier, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * (num_layers - 1) + [out_size]
        self.vae_model = vae_model
        self.layers = nn.ModuleList(
            [nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])
        self.relu = nn.ReLU()

    def forward(self, inputs, lengths):
        _, latent = self.vae_model(inputs, lengths)
        latent = torch.transpose(latent[0], 1, 2)

        for i, layer in enumerate(self.layers[:-1]):
            latent = self.relu(layer(latent))
        latent = self.layers[-1](latent)

        return torch.transpose(latent, 1, 2)


class curlEncodedClassifier(nn.Module):

    def __init__(self, curl_model, input_size, num_layers, hidden_size, out_size):
        super(curlEncodedClassifier, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * (num_layers - 1) + [out_size]
        self.curl_model = curl_model
        self.layers = nn.ModuleList(
            [nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1) for (in_size, out_size)
             in
             zip(input_sizes, output_sizes)])
        self.relu = nn.ReLU()

    def forward(self, inputs, lengths):
        _, latent = self.curl_model(inputs, lengths)
        latent = compute_latent_features(latent)
        latent = torch.transpose(latent, 1, 2)

        for i, layer in enumerate(self.layers[:-1]):
            latent = self.relu(layer(latent))
        latent = self.layers[-1](latent)

        return torch.transpose(latent, 1, 2)


class curlEncoder(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, bn_size, comp_num):
        super(curlEncoder, self).__init__()

        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])
        self.comp_num = comp_num

        self.means = nn.ModuleList(
            [nn.Linear(in_features=hidden_size, out_features=bn_size) for i in
             range(comp_num)])

        self.var = nn.ModuleList(
            [nn.Linear(in_features=hidden_size, out_features=bn_size) for i in
             range(comp_num)])

        self.categorical = nn.Linear(in_features=hidden_size, out_features=comp_num)
        self.sm = nn.Softmax(dim=2)

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):
            inputs, _ = layer(inputs)

        inputs, _ = pad_packed_sequence(inputs, True, total_length=seq_len)

        return self.sm(self.categorical(inputs)), torch.cat(
            [self.means[i](inputs)[None, :] for i in range(self.comp_num)]), torch.cat(
            [self.var[i](inputs)[None, :] for i in range(self.comp_num)])


class curlDecoder(nn.Module):
    def __init__(self, bn_size, num_layers, hidden_size, input_size):
        super(curlDecoder, self).__init__()

        input_sizes = [bn_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.means = nn.Linear(in_features=hidden_size, out_features=input_size)

    def forward(self, all_inputs, lengths):
        outs = []
        for inputs in all_inputs:
            seq_len = inputs.size(1)
            inputs = pack_padded_sequence(inputs, lengths, True)

            for i, layer in enumerate(self.layers):
                inputs, _ = layer(inputs)

            inputs, _ = pad_packed_sequence(inputs, True, total_length=seq_len)

            means = self.means(inputs)
            outs.append(means[None, :])
        return torch.cat(outs)


class curlDecoderMultistream(nn.Module):
    def __init__(self, bn_size, num_streams, num_layers, hidden_size, input_size):
        super(curlDecoderMultistream, self).__init__()

        input_sizes = [bn_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList([nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)]) for i in range(num_streams)])

        self.means = nn.ModuleList(
            [nn.Linear(in_features=hidden_size, out_features=input_size) for i in range(num_streams)])

    def forward(self, all_inputs, lengths):
        outs = []
        for idx, inputs in enumerate(all_inputs):
            seq_len = inputs.size(1)
            inputs = pack_padded_sequence(inputs, lengths, True)

            for i, layer in enumerate(self.layers[idx]):
                inputs, _ = layer(inputs)

            inputs, _ = pad_packed_sequence(inputs, True, total_length=seq_len)

            means = self.means[idx](inputs)
            outs.append(means[None, :])
        return torch.cat(outs)


class curlLatentSampler(nn.Module):
    def __init__(self, use_gpu=True):
        super(curlLatentSampler, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, latent):
        sampled = []
        for idx, m in enumerate(latent[1]):
            v = latent[2][idx]
            if self.use_gpu:
                sampled.append((m + torch.exp(v) * torch.randn(m.shape).cuda())[None, :])
            else:
                sampled.append((m + torch.exp(v) * torch.randn(m.shape))[None, :])

        return torch.cat(sampled)


class nnetCurlSupervised(nn.Module):
    def __init__(self, input_size, num_layers_enc, num_layers_dec, hidden_size, bn_size, comp_num, use_gpu=True):
        super(nnetCurlSupervised, self).__init__()

        self.curl_encoder = curlEncoder(input_size, num_layers_enc, hidden_size, bn_size, comp_num)
        self.curl_decoder = curlDecoder(bn_size, num_layers_dec, hidden_size, input_size)
        self.curl_sampler = curlLatentSampler(use_gpu)

    def forward(self, inputs, lengths):
        latent = self.curl_encoder(inputs, lengths)
        sampled = self.curl_sampler(latent)
        return self.curl_decoder(sampled, lengths), latent


class nnetCurlMultistreamClassifier(nn.Module):
    def __init__(self, input_size, num_layers_enc, num_layers_dec, num_layers_class, hidden_size,
                 hidden_size_classifier, bn_size, comp_num, out_size, use_gpu=True, enc_scale=0.2):
        super(nnetCurlMultistreamClassifier, self).__init__()

        self.comp_num = comp_num
        self.input_size = input_size
        self.out_size = out_size
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.num_layers_class = num_layers_class
        self.hidden_size = hidden_size
        self.hidden_size_classifier = hidden_size_classifier
        self.enc_scale = enc_scale
        self.bn_size = bn_size

        self.curl_encoder = curlEncoder(input_size, num_layers_enc, hidden_size, bn_size, comp_num)
        self.curl_decoder = curlDecoderMultistream(bn_size, comp_num, num_layers_dec, hidden_size, input_size)
        self.curl_sampler = curlLatentSampler(use_gpu)

        self.classifier = nn.ModuleList(
            [decoderRNN(bn_size, num_layers_class, hidden_size_classifier, out_size) for i in range(self.comp_num)])
        self.use_gpu = use_gpu

    def expand_component(self, use_gpu):
        self.comp_num = self.comp_num + 1

        if use_gpu:
            # Add extra Gaussian
            self.curl_encoder.means.append(nn.Linear(in_features=self.hidden_size, out_features=self.bn_size).cuda())
            self.curl_encoder.var.append(nn.Linear(in_features=self.hidden_size, out_features=self.bn_size).cuda())
            updated_y = nn.Linear(in_features=self.hidden_size, out_features=self.comp_num).cuda()
            with torch.no_grad():
                updated_y.weight[0:self.comp_num - 1, :] = self.curl_encoder.categorical.weight
                updated_y.weight[0:self.comp_num - 1] = self.curl_encoder.categorical.bias
            self.curl_encoder.categorical = updated_y
        else:
            # Add extra Gaussian
            self.curl_encoder.means.append(nn.Linear(in_features=self.hidden_size, out_features=self.bn_size))
            self.curl_encoder.var.append(nn.Linear(in_features=self.hidden_size, out_features=self.bn_size))
            updated_y = nn.Linear(in_features=self.hidden_size, out_features=self.comp_num)
            with torch.no_grad():
                updated_y.weight[0:self.comp_num - 1, :] = self.curl_encoder.categorical.weight
                updated_y.weight[0:self.comp_num - 1] = self.curl_encoder.categorical.bias
            self.curl_encoder.categorical = updated_y

        self.curl_encoder.comp_num = self.comp_num

        # Add extra reconstruction and classifier decoders
        input_sizes = [self.bn_size] + [self.hidden_size] * (self.num_layers_dec - 1)
        output_sizes = [self.hidden_size] * self.num_layers_dec
        self.curl_decoder.layers.append(nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)]))
        self.curl_decoder.means.append(nn.Linear(in_features=self.hidden_size, out_features=self.input_size))

        self.classifier.append(
            decoderRNN(self.bn_size, self.num_layers_class, self.hidden_size_classifier, self.out_size))

    def forward(self, inputs, lengths):
        latent = self.curl_encoder(inputs, lengths)
        # Scale the gradients

        h1 = latent[0].register_hook(lambda grad: grad * self.enc_scale)
        h2 = latent[1].register_hook(lambda grad: grad * self.enc_scale)
        h3 = latent[2].register_hook(lambda grad: grad * self.enc_scale)

        sampled_latent = self.curl_sampler(latent)
        class_out = []

        for idx, stream in enumerate(self.classifier):
            class_out.append(stream(sampled_latent[idx], lengths))

        return class_out, self.curl_decoder(sampled_latent, lengths), latent


def compute_latent_features(latent, use_gpu=True):
    latent = list(latent)
    # embeddings = latent[1]
    # prior = latent[0]
    s = latent[1].shape
    latent[1] = latent[1].view(s[0], s[1] * s[2], s[3])
    latent[0] = latent[0].view(s[1] * s[2], -1)
    feats = torch.zeros(s[1] * s[2], s[3])
    if use_gpu:
        feats = feats.cuda()
    for idx in range(latent[0].shape[1]):
        feats += latent[1][idx, :] * torch.cat(s[3] * [(latent[0][:, idx])[:, None]], dim=1)

    return feats.view(s[1], s[2], s[3])


class modnetEncoder(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, input_filter_kernel, freq_num, wind_size,
                 head_num,
                 init_mod=True, use_gpu=True):
        super(modnetEncoder, self).__init__()

        layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            layers.append(nn.Conv2d(in_size, out_size, kernel))
            input_h -= (kernel - 1)
            input_w -= (kernel - 1)
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.cnn_out_dim = out_channels[-1] * input_h * input_w
        self.input_filter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=input_filter_kernel, stride=1,
                                      padding=int(
                                          np.floor(input_filter_kernel / 2)))  # Simple filter to smoothen inputs

        regressors = []
        for i in range(head_num):
            lin = nn.Linear(out_channels[-1] * input_h * input_w, freq_num)
            if init_mod:
                if i < freq_num:  # initialize the head weights
                    with torch.no_grad():
                        init_mat = torch.rand(freq_num, out_channels[-1] * input_h * input_w)
                        init_mat[i, :] = 1
                        lin.weight.copy_(init_mat)
                        lin.bias.copy_(torch.FloatTensor(torch.rand(freq_num)))

            regressors.append(lin)

        self.regressors = nn.ModuleList(regressors)
        self.gumbel_sm = gumbel_softmax
        self.wind_size = wind_size
        self.freq_num = freq_num
        self.use_gpu = use_gpu

    def forward(self, inputs):

        # Filter the input with long context 1D filter
        # feats = [(self.input_filter(inputs[:, :, inx, :]))[:, :, None, :] for inx in range(inputs.shape[2])]
        # feats = torch.cat(feats, dim=2)
        feats = inputs
        s = feats.shape  # batch_num x in_channels x height x width

        for i, layer in enumerate(self.layers):
            inputs = self.relu(layer(inputs))

        if self.use_gpu:
            fs = Variable(1 / self.wind_size * torch.linspace(1, self.freq_num, self.freq_num)).cuda()
            fs = torch.cat(s[0] * [fs[None, :]])
        else:
            fs = Variable(1 / self.wind_size * torch.linspace(1, self.freq_num, self.freq_num))
            fs = torch.cat(s[0] * [fs[None, :]])

        modulations = []
        mod_f = []
        for regressor in self.regressors:
            f = torch.sum(self.gumbel_sm(regressor(inputs.view(-1, self.cnn_out_dim)), 0.8) * fs, dim=1)
            mod_f.append(f[:, None])
            f = torch.cat(s[3] * [f[:, None]], dim=1)  # batch x w
            if self.use_gpu:
                t = torch.linspace(0, self.wind_size, s[3]).cuda()
            else:
                t = torch.linspace(0, self.wind_size, s[3])

            t = torch.cat(s[0] * [t[None, :]], dim=0)  # batch x w
            sins = torch.sin(2 * np.pi * f * t)
            sins = torch.cat(s[2] * [sins[:, None, :]], dim=1)  # batch x height x width
            modulations.append(torch.mean(sins * feats[:, 0, :, :], dim=2))  # batch x height
        return torch.cat(modulations, dim=1), torch.cat(mod_f, dim=1)


class modnetClassifier(nn.Module):
    def __init__(self, input_size, out_size, num_layers, hidden_size):
        super(modnetClassifier, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * (num_layers - 1) + [out_size]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for (in_size, out_size) in zip(input_sizes, output_sizes)])
        self.relu = nn.ReLU()

    def forward(self, input):
        for layer in self.layers[:-1]:
            input = self.relu(layer(input))
        input = self.layers[-1](input)

        return input


class modulationNet(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, input_filter_kernel, freq_num, wind_size,
                 head_num,
                 num_layers_dec, hidden_size, out_size, init_mod, use_gpu):
        super(modulationNet, self).__init__()

        self.encoder = modnetEncoder(input_h, input_w, in_channels, out_channels, kernel, input_filter_kernel, freq_num,
                                     wind_size,
                                     head_num, init_mod, use_gpu)
        self.classifier = modnetClassifier(input_h * head_num, out_size, num_layers_dec, hidden_size)

    def forward(self, input):
        input, mod_f = self.encoder(input)
        input = self.classifier(input)
        return input, mod_f


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class modnetSigmoidEncoder(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, input_filter_kernel, freq_num, wind_size,
                 use_gpu=True):
        super(modnetSigmoidEncoder, self).__init__()

        layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            layers.append(nn.Conv2d(in_size, out_size, kernel))
            input_h -= (kernel - 1)
            input_w -= (kernel - 1)
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.cnn_out_dim = out_channels[-1] * input_h * input_w
        self.input_filter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=input_filter_kernel, stride=1,
                                      padding=int(
                                          np.floor(input_filter_kernel / 2)))  # Simple filter to smoothen inputs

        self.regression = nn.Linear(out_channels[-1] * input_h * input_w, freq_num)
        self.sigmoid = nn.Sigmoid()
        self.wind_size = wind_size
        self.freq_num = freq_num
        self.use_gpu = use_gpu

    def forward(self, inputs):

        # Get filtered features
        feats = [(self.input_filter(inputs[:, :, inx, :]))[:, :, None, :] for inx in range(inputs.shape[2])]
        feats = torch.cat(feats, dim=2)
        s = feats.shape  # batch_num x in_channels x height x width

        # Get weights for modulation features
        for i, layer in enumerate(self.layers):
            inputs = self.relu(layer(inputs))
        inputs = self.sigmoid(self.regression(inputs.view(-1, self.cnn_out_dim)))  # batch_num x freq_num

        if self.use_gpu:
            fs = Variable(1 / self.wind_size * torch.linspace(1, self.freq_num, self.freq_num)).cuda()
        else:
            fs = Variable(1 / self.wind_size * torch.linspace(1, self.freq_num, self.freq_num))

        # Mean weighted modulation
        wtd_mean_mod = torch.mean(inputs * torch.cat(s[0] * [fs[None, :]], dim=0))
        modulations = []

        # Create the sinusoid tensor
        for idx in range(self.freq_num):
            if self.use_gpu:
                f = fs[idx] * torch.ones(s[0], s[2], s[3]).cuda()  # batch x h x w
                t = torch.linspace(0, self.wind_size, s[3]).cuda()
            else:
                f = fs[idx] * torch.ones(s[0], s[2], s[3])  # batch x h x w
                t = torch.linspace(0, self.wind_size, s[3])

            t = torch.cat(s[0] * [t[None, :]], dim=0)  # batch x w
            t = torch.cat(s[2] * [t[:, None, :]], dim=1)  # batch x h x w
            sins = torch.mean(torch.sin(2 * np.pi * f * t) * feats[:, 0, :, :], dim=2)  # batch x height
            coss = torch.mean(torch.cos(2 * np.pi * f * t) * feats[:, 0, :, :], dim=2)  # batch x height
            sins = torch.sqrt(torch.pow(sins, 2) + torch.pow(coss, 2))
            wts = torch.cat(s[2] * [(inputs[:, idx])[:, None]], dim=1)  # batch x height
            modulations.append(sins * wts)  # batch x height

        return torch.cat(modulations, dim=1), wtd_mean_mod


class modulationSigmoidNet(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, input_filter_kernel, freq_num, wind_size,
                 num_layers_dec, hidden_size, out_size, use_gpu):
        super(modulationSigmoidNet, self).__init__()

        self.encoder = modnetSigmoidEncoder(input_h, input_w, in_channels, out_channels, kernel, input_filter_kernel,
                                            freq_num, wind_size,
                                            use_gpu)
        self.classifier = modnetClassifier(input_h * freq_num, out_size, num_layers_dec, hidden_size)

    def forward(self, input):
        input, mod_f = self.encoder(input)
        input = self.classifier(input)
        return input, mod_f


class cnnClassifier(nn.Module):

    def __init__(self, input_h, input_w, in_channels, out_channels, kernel,
                 num_layers_dec, hidden_size, output_size):
        super(cnnClassifier, self).__init__()

        layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            layers.append(nn.Conv2d(in_size, out_size, kernel))
            input_h -= (kernel - 1)
            input_w -= (kernel - 1)

        self.cnn_layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.cnn_out_dim = out_channels[-1] * input_h * input_w

        input_sizes = [self.cnn_out_dim] + [hidden_size] * (num_layers_dec - 1)
        output_sizes = [hidden_size] * (num_layers_dec - 1) + [output_size]

        self.linear_layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for (in_size, out_size) in zip(input_sizes, output_sizes)])

    def forward(self, inputs):

        for layer in self.cnn_layers:
            inputs = self.relu(layer(inputs))

        inputs = inputs.view(-1, self.cnn_out_dim)

        for layer in self.linear_layers[:-1]:
            inputs = self.relu(layer(inputs))

        inputs = self.linear_layers[-1](inputs)
        return inputs

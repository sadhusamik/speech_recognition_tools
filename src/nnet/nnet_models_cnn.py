import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class nnetCNNClassifier(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, output_size):
        super(nnetCNNClassifier, self).__init__()
        layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            layers.append(nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                    padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))

        self.cnn_layers = nn.ModuleList(layers)
        self.cnn_out_dim = out_channels[-1] * input_h
        self.input_h = input_h
        self.input_w = input_w
        self.relu = nn.ReLU()
        self.lin = nn.Conv1d(in_channels=self.cnn_out_dim, out_channels=output_size, kernel_size=1, stride=1)

    def forward(self, inputs):

        for idx, layer in enumerate(self.cnn_layers):
            inputs = self.relu(layer(inputs))

        inputs = inputs.view(inputs.shape[0], -1, inputs.shape[3])  # batch x sort of freq x time
        return torch.transpose(self.lin(inputs), 1, 2)


class nnetCLDNN(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, hidden_size, l_num_layers, d_num_layers,
                 output_size):
        super(nnetCLDNN, self).__init__()

        cnn_layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            cnn_layers.append(nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                        padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))
        cnn_layers = nn.ModuleList(cnn_layers)
        self.cnn_layers = cnn_layers

        self.lstm_layers = nn.ModuleList(
            [nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True) for i in range(l_num_layers)])

        input_sizes = [hidden_size] * d_num_layers
        output_sizes = [hidden_size] * (d_num_layers - 1) + [output_size]

        self.dnn_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.cnn_out_dim = out_channels[-1] * input_h
        self.input_h = input_h
        self.input_w = input_w
        self.relu = nn.ReLU()
        self.dim_reduce = nn.Conv1d(in_channels=self.cnn_out_dim, out_channels=hidden_size, kernel_size=1, stride=1)

    def forward(self, inputs, lengths):

        for idx, layer in enumerate(self.cnn_layers):
            inputs = self.relu(layer(inputs))

        inputs = inputs.view(inputs.shape[0], -1, inputs.shape[3])  # batch x sort of freq x time
        inputs = torch.transpose(self.dim_reduce(inputs), 1, 2)

        seq_len = inputs.size(1)
        inputs = pack_padded_sequence(inputs, lengths, True)

        for idx, layer in enumerate(self.lstm_layers):
            inputs, _ = layer(inputs)

        inputs, _ = pad_packed_sequence(inputs, True, total_length=seq_len)

        inputs = torch.transpose(inputs, 1, 2)
        for idx, layer in enumerate(self.dnn_layers[:-1]):
            inputs = self.relu(layer(inputs))

        inputs = self.dnn_layers[-1](inputs)

        return torch.transpose(inputs, 1, 2)


class nnetCLDNN3D(nn.Module):
    def __init__(self, input_h, input_w, num_streams, in_channels, out_channels, kernel, hidden_size, l_num_layers,
                 d_num_layers,
                 output_size):
        super(nnetCLDNN3D, self).__init__()

        all_streams_cnn = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            cnn_layers = []
            for i in range(num_streams):
                cnn_layers.append(nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                            padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))
            cnn_layers = nn.ModuleList(cnn_layers)
        all_streams_cnn.append(cnn_layers)

        all_streams_cnn = nn.ModuleList(all_streams_cnn)
        self.cnn_layers = all_streams_cnn

        self.lstm_layers = nn.ModuleList(
            [nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True) for i in range(l_num_layers)])

        input_sizes = [hidden_size] * d_num_layers
        output_sizes = [hidden_size] * (d_num_layers - 1) + [output_size]

        self.dnn_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.cnn_out_dim = out_channels[-1] * input_h * num_streams
        self.input_h = input_h
        self.input_w = input_w
        self.num_streams = num_streams
        self.relu = nn.ReLU()
        self.dim_reduce = nn.Conv1d(in_channels=self.cnn_out_dim, out_channels=hidden_size, kernel_size=1, stride=1)

    def forward(self, inputs, lengths):

        # inputs are batch x in_channels x num_streams x h x w
        for idx, filts in enumerate(self.cnn_layers):
            for idx2, layer in enumerate(filts):
                inputs[:, :, idx2, :, :] = self.relu(layer(inputs[:, :, idx2, :, :]))
        inputs = inputs.view(inputs.shape[0], -1, inputs.shape[4])  # batch x sort of freq x time
        inputs = torch.transpose(self.dim_reduce(inputs), 1, 2)
        seq_len = inputs.size(1)
        inputs = pack_padded_sequence(inputs, lengths, True)

        for idx, layer in enumerate(self.lstm_layers):
            inputs, _ = layer(inputs)

        inputs, _ = pad_packed_sequence(inputs, True, total_length=seq_len)

        inputs = torch.transpose(inputs, 1, 2)
        for idx, layer in enumerate(self.dnn_layers[:-1]):
            inputs = self.relu(layer(inputs))

        inputs = self.dnn_layers[-1](inputs)

        return torch.transpose(inputs, 1, 2)


class VAECNNEncoder(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, bn_size):
        super(VAECNNEncoder, self).__init__()

        layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            layers.append(nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                    padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))
            input_h = int(np.floor((input_h - 2) / 2 + 1))
            input_w = int(np.floor((input_w - 2) / 2 + 1))

        self.cnn_layers = nn.ModuleList(layers)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.cnn_out_dim = out_channels[-1] * input_h * input_w
        self.input_h = input_h
        self.input_w = input_w
        self.relu = nn.ReLU()

        self.means = nn.Linear(in_features=self.cnn_out_dim, out_features=bn_size)
        self.vars = nn.Linear(in_features=self.cnn_out_dim, out_features=bn_size)

    def forward(self, inputs):
        indices = []
        sizes = []
        for idx, layer in enumerate(self.cnn_layers):
            inputs = self.relu(layer(inputs))
            sizes.append(inputs.shape)
            inputs, id = self.mp(inputs)
            indices.append(id)

        inputs = inputs.view(inputs.shape[0], -1)

        return self.means(inputs), self.vars(inputs), indices, sizes


class VAECNNEncoderNopool(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, bn_size):
        super(VAECNNEncoderNopool, self).__init__()

        layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            layers.append(nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                    padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))

        self.cnn_layers = nn.ModuleList(layers)
        self.cnn_out_dim = out_channels[-1] * input_h
        self.input_h = input_h
        self.input_w = input_w
        self.relu = nn.ReLU()

        self.means = nn.Conv1d(in_channels=self.cnn_out_dim, out_channels=bn_size, kernel_size=1, stride=1)
        self.vars = nn.Conv1d(in_channels=self.cnn_out_dim, out_channels=bn_size, kernel_size=1, stride=1)

    def forward(self, inputs):

        for idx, layer in enumerate(self.cnn_layers):
            inputs = self.relu(layer(inputs))

        w_change = inputs.shape[3]
        inputs_save = inputs
        inputs = inputs.view(inputs.shape[0], -1, inputs.shape[3])  # batch x sort of freq x time

        return self.means(inputs), self.vars(inputs), w_change, inputs_save


class VAECNNDecoder(nn.Module):
    def __init__(self, input_h, input_w, bn_size, in_channels, out_channels, kernel):
        super(VAECNNDecoder, self).__init__()

        self.expand_linear = nn.Linear(in_features=bn_size, out_features=input_h * input_w * in_channels[0])
        self.expand_size = in_channels[0]
        self.input_h = input_h
        self.input_w = input_w

        layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            layers.append(nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                             padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))

        self.cnn_layers = nn.ModuleList(layers)
        self.mup = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, inputs, maxpool_indices, sizes):

        inputs = self.expand_linear(inputs)
        inputs = inputs.view(inputs.size(0), self.expand_size, self.input_h, self.input_w)

        for idx, layer in enumerate(self.cnn_layers[:-1]):
            inputs = self.relu(layer(self.mup(inputs, maxpool_indices[idx], output_size=sizes[idx])))

        idx = len(self.cnn_layers) - 1
        inputs = self.cnn_layers[-1](self.mup(inputs, maxpool_indices[idx], output_size=sizes[idx]))

        return inputs


class VAECNNDecoderNopool(nn.Module):
    def __init__(self, input_h, input_w, bn_size, in_channels, out_channels, kernel):
        super(VAECNNDecoderNopool, self).__init__()

        self.expand_linear = nn.Conv1d(in_channels=bn_size, out_channels=input_h * in_channels[0], kernel_size=1,
                                       stride=1)
        self.expand_size = in_channels[0]
        self.input_h = input_h
        self.input_w = input_w

        layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            layers.append(nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                             padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))

        self.cnn_layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, inputs, input_w):

        inputs = self.expand_linear(inputs)
        inputs = inputs.view(inputs.size(0), self.expand_size, self.input_h, input_w)

        for idx, layer in enumerate(self.cnn_layers[:-1]):
            inputs = self.relu(layer(inputs))

        inputs = self.cnn_layers[-1](inputs)

        return inputs


class latentSamplerCNN(nn.Module):
    def __init__(self, use_gpu=True):
        super(latentSamplerCNN, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, latent):
        if self.use_gpu:
            return latent[0] + torch.exp(latent[1]) * torch.randn(latent[0].shape).cuda()
        else:

            return latent[0] + torch.exp(latent[1]) * torch.randn(latent[0].shape)


class nnetVAECNN(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, bn_size, use_gpu=True):
        super(nnetVAECNN, self).__init__()

        self.vae_encoder = VAECNNEncoder(input_h, input_w, in_channels, out_channels, kernel, bn_size)
        self.vae_decoder = VAECNNDecoder(self.vae_encoder.input_h, self.vae_encoder.input_w, bn_size,
                                         out_channels[::-1],
                                         in_channels[::-1], kernel)
        self.sampler = latentSamplerCNN(use_gpu)

    def forward(self, inputs):
        latent = self.vae_encoder(inputs)
        inputs = self.sampler(latent)
        return self.vae_decoder(inputs, latent[2][::-1], latent[3][::-1]), latent


class nnetVAECNNNopool(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, bn_size, use_gpu=True):
        super(nnetVAECNNNopool, self).__init__()

        self.vae_encoder = VAECNNEncoderNopool(input_h, input_w, in_channels, out_channels, kernel, bn_size)
        self.vae_decoder = VAECNNDecoderNopool(self.vae_encoder.input_h, self.vae_encoder.input_w, bn_size,
                                               out_channels[::-1],
                                               in_channels[::-1], kernel)
        self.sampler = latentSamplerCNN(use_gpu)

    def forward(self, inputs):
        latent = self.vae_encoder(inputs)
        inputs = self.sampler(latent)
        return self.vae_decoder(inputs, latent[2]), latent


class VAECNNEncoderNopoolAE(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, bn_size):
        super(VAECNNEncoderNopoolAE, self).__init__()

        layers = []
        for (in_size, out_size) in zip(in_channels, out_channels):
            layers.append(nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                    padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))

        self.cnn_layers = nn.ModuleList(layers)
        self.cnn_out_dim = out_channels[-1] * input_h
        self.input_h = input_h
        self.input_w = input_w
        self.relu = nn.ReLU()

        self.bn = nn.Conv1d(in_channels=self.cnn_out_dim, out_channels=bn_size, kernel_size=1, stride=1)

    def forward(self, inputs):

        for idx, layer in enumerate(self.cnn_layers):
            inputs = self.relu(layer(inputs))

        w_change = inputs.shape[3]
        inputs_save = inputs
        inputs = inputs.view(inputs.shape[0], -1, inputs.shape[3])  # batch x sort of freq x time

        return self.relu(self.bn(inputs)), w_change, inputs_save


class nnetCNNAE(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, bn_size):
        super(nnetCNNAE, self).__init__()
        self.vae_encoder = VAECNNEncoderNopoolAE(input_h, input_w, in_channels, out_channels, kernel, bn_size)
        self.vae_decoder = VAECNNDecoderNopool(self.vae_encoder.input_h, self.vae_encoder.input_w, bn_size,
                                               out_channels[::-1],
                                               in_channels[::-1], kernel)

    def forward(self, inputs):
        latent = self.vae_encoder(inputs)
        return self.vae_decoder(latent[0], latent[1]), latent


class rsconv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, use_gpu=False):
        super(rsconv2d, self).__init__()

        self.rates = torch.nn.Parameter(torch.rand(out_channel, in_channel))
        self.scales = torch.nn.Parameter(torch.zeros(out_channel, in_channel))
        self.rates.requires_grad = True
        self.scales.requires_grad = True

        self.in_c = in_channel
        self.out_c = out_channel
        self.kernel_size = kernel_size
        self.padding = padding

        self.use_gpu = use_gpu

        t = np.arange(kernel_size[1])
        f = np.arange(kernel_size[0])
        X = np.meshgrid(t, f)
        WW = np.outer(np.hanning(kernel_size[0]), np.hanning(kernel_size[1]))
        if use_gpu:
            self.mesh = [torch.from_numpy(X[0]).float().cuda(), torch.from_numpy(X[1]).float().cuda()]
            self.WW = torch.from_numpy(WW).float().cuda()
        else:
            self.mesh = [torch.from_numpy(X[0]).float(), torch.from_numpy(X[1]).float()]
            self.WW = torch.from_numpy(WW).float()

    def forward(self, inputs):

        if self.use_gpu:
            weights = torch.zeros(self.out_c, self.in_c, self.kernel_size[0], self.kernel_size[1]).cuda()
        else:
            weights = torch.zeros(self.out_c, self.in_c, self.kernel_size[0], self.kernel_size[1])

        for i in range(self.out_c):
            for j in range(self.in_c):
                weights[i, j] = torch.sin(self.rates[i, j] * self.mesh[0] + self.scales[i, j] * self.mesh[1]) * self.WW

        return F.conv2d(inputs, weights, bias=None, stride=1, padding=self.padding, dilation=1, groups=1)


class rsconvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, use_gpu=False):
        super(rsconvTranspose2d, self).__init__()

        self.rates = torch.nn.Parameter(torch.rand(in_channel, out_channel))
        self.scales = torch.nn.Parameter(torch.zeros(in_channel, out_channel))
        self.rates.requires_grad = True
        self.scales.requires_grad = True

        self.in_c = in_channel
        self.out_c = out_channel
        self.kernel_size = kernel_size
        self.padding = padding

        self.use_gpu = use_gpu

        t = np.arange(kernel_size[1])
        f = np.arange(kernel_size[0])
        X = np.meshgrid(t, f)
        WW = np.outer(np.hanning(kernel_size[0]), np.hanning(kernel_size[1]))
        if use_gpu:
            self.mesh = [torch.from_numpy(X[0]).float().cuda(), torch.from_numpy(X[1]).float().cuda()]
            self.WW = torch.from_numpy(WW).float().cuda()
        else:
            self.mesh = [torch.from_numpy(X[0]).float(), torch.from_numpy(X[1]).float()]
            self.WW = torch.from_numpy(WW).float()

    def forward(self, inputs):

        if self.use_gpu:
            weights = torch.zeros(self.in_c, self.out_c, self.kernel_size[0], self.kernel_size[1]).cuda()
        else:
            weights = torch.zeros(self.in_c, self.out_c, self.kernel_size[0], self.kernel_size[1])

        for i in range(self.in_c):
            for j in range(self.out_c):
                weights[i, j] = torch.sin(self.rates[i, j] * self.mesh[0] + self.scales[i, j] * self.mesh[1]) * self.WW

        return F.conv_transpose2d(inputs, weights, bias=None, stride=1, padding=self.padding, dilation=1, groups=1)


class VAECNNModulationEncoder(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, bn_size, use_gpu):
        super(VAECNNModulationEncoder, self).__init__()

        layers = []
        for (in_size, out_size) in zip(in_channels[:-1], out_channels[:-1]):
            layers.append(nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                    padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))

        # Final layer make it a modulation one
        layers.append(
            rsconv2d(in_channels[-1], out_channels[-1], kernel, (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2)),
                     use_gpu))

        self.cnn_layers = nn.ModuleList(layers)
        self.cnn_out_dim = out_channels[-1] * input_h
        self.input_h = input_h
        self.input_w = input_w
        self.relu = nn.ReLU()

        self.means = nn.Conv1d(in_channels=self.cnn_out_dim, out_channels=bn_size, kernel_size=1, stride=1)
        self.vars = nn.Conv1d(in_channels=self.cnn_out_dim, out_channels=bn_size, kernel_size=1, stride=1)

    def forward(self, inputs):
        for idx, layer in enumerate(self.cnn_layers):
            inputs = self.relu(layer(inputs))

        w_change = inputs.shape[3]
        inputs = inputs.view(inputs.shape[0], -1, inputs.shape[3])  # batch x sort of freq x time

        return self.means(inputs), self.vars(inputs), w_change


class VAECNNModulationDecoder(nn.Module):
    def __init__(self, input_h, input_w, bn_size, in_channels, out_channels, kernel, use_gpu):
        super(VAECNNModulationDecoder, self).__init__()

        self.expand_linear = nn.Conv1d(in_channels=bn_size, out_channels=input_h * in_channels[0], kernel_size=1,
                                       stride=1)
        self.expand_size = in_channels[0]
        self.input_h = input_h
        self.input_w = input_w

        layers = []

        layers.append(rsconvTranspose2d(in_channels[0], out_channels[0], kernel,
                                        (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2)), use_gpu))

        for (in_size, out_size) in zip(in_channels[1:], out_channels[1:]):
            layers.append(nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel,
                                             padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))))

        self.cnn_layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, inputs, input_w):

        inputs = self.expand_linear(inputs)
        inputs = inputs.view(inputs.size(0), self.expand_size, self.input_h, input_w)

        for idx, layer in enumerate(self.cnn_layers[:-1]):
            inputs = self.relu(layer(inputs))

        inputs = self.cnn_layers[-1](inputs)

        return inputs


class nnetVaeRsModulation(nn.Module):
    def __init__(self, input_h, input_w, in_channels, out_channels, kernel, bn_size, use_gpu=True):
        super(nnetVaeRsModulation, self).__init__()

        self.vae_encoder = VAECNNModulationEncoder(input_h, input_w, in_channels, out_channels, kernel, bn_size,
                                                   use_gpu)
        self.vae_decoder = VAECNNModulationDecoder(self.vae_encoder.input_h, self.vae_encoder.input_w, bn_size,
                                                   out_channels[::-1],
                                                   in_channels[::-1], kernel, use_gpu)
        self.sampler = latentSamplerCNN(use_gpu)

    def forward(self, inputs):
        latent = self.vae_encoder(inputs)
        inputs = self.sampler(latent)
        return self.vae_decoder(inputs, latent[2]), latent

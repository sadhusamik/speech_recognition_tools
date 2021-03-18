import os
import argparse
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from nnet_models import nnetVAE, nnetRNN, nnetLinearWithConv, nnetARVAE
from nnet_models_cnn import nnetVAECNNNopool
import kaldi_io
from features import dict2Ark
import numpy as np
import pickle
import sys

def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def get_args():
    parser = argparse.ArgumentParser(description="Dump likelihoods or posteriors from VAE encoded model")

    parser.add_argument("model", help="Pytorch nnet model")
    parser.add_argument("scp", help="scp files for features")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("save_file", help="file to save posteriors")
    parser.add_argument("--vae_arch", default="cnn", help="Type of VAE layers CNN or RNN")
    parser.add_argument("--max_seq_len", default=512, type=int, help="Maximum sequence length for CNN VAE")
    parser.add_argument("--prior", default=None, help="Provide prior to normalize and get likelihoods")
    parser.add_argument("--prior_weight", type=float, default=0.8, help="Weight for the prior distribution")
    parser.add_argument("--add_softmax", action="store_true", help="Set to add softmax when dumping posteriors")
    parser.add_argument("--override_trans", default=None, help="Provide a different feature transformation file")
    return parser.parse_args()


def get_output(config):
    # Load model
    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    vae = torch.load(nnet['vaeenc'], map_location=lambda storage, loc: storage)

    if nnet['vae_type'] == "modulation":
        if config.vae_arch == "cnn":
            in_channels = [int(x) for x in vae['in_channels'].split(',')]
            out_channels = [int(x) for x in vae['out_channels'].split(',')]
            kernel = tuple([int(x) for x in vae['kernel'].split(',')])
            vae_model = nnetVAECNNNopool(vae['feature_dim'], vae['num_frames'], in_channels,
                                         out_channels, kernel, vae['nfilters'] * vae['nrepeats'], False)
        else:
            vae_model = nnetVAE(vae['feature_dim'] * vae['num_frames'], vae['encoder_num_layers'],
                                vae['decoder_num_layers'], vae['hidden_dim'], vae['nfilters'] * vae['nrepeats'], 0,
                                False)
        model = nnetRNN(vae['nfilters'] * vae['nrepeats'], nnet['num_layers'], nnet['hidden_dim'],
                        nnet['num_classes'], 0)
        vae_model.load_state_dict(vae["model_state_dict"])
        model.load_state_dict(nnet["model_state_dict"])

    elif nnet['vae_type'] == "arvae":
        ar_steps = vae['ar_steps'].split(',')
        ar_steps = [int(x) for x in ar_steps]
        ar_steps.append(0)
        vae_model = nnetARVAE(vae['feature_dim'] * vae['num_frames'], vae['encoder_num_layers'],
                              vae['decoder_num_layers'], vae['hidden_dim'], vae['bn_dim'], 0, len(ar_steps),
                              False)
        model = nnetRNN(vae['bn_dim'], nnet['num_layers'], nnet['hidden_dim'],
                        nnet['num_classes'], 0)
        vae_model.load_state_dict(vae["model_state_dict"])
        model.load_state_dict(nnet['model_state_dict'])
    else:
        if config.vae_arch == "cnn":
            in_channels = [int(x) for x in vae['in_channels'].split(',')]
            out_channels = [int(x) for x in vae['out_channels'].split(',')]
            kernel = tuple([int(x) for x in vae['kernel'].split(',')])
            vae_model = nnetVAECNNNopool(vae['feature_dim'], vae['num_frames'], in_channels,
                                         out_channels, kernel, vae['bn_dim'], False)
        else:
            vae_model = nnetVAE(vae['feature_dim'] * vae['num_frames'], vae['encoder_num_layers'],
                                vae['decoder_num_layers'], vae['hidden_dim'], vae['bn_dim'], 0,
                                False)

        model = nnetRNN(vae['bn_dim'], nnet['num_layers'], nnet['hidden_dim'],
                        nnet['num_classes'], 0)
        vae_model.load_state_dict(vae["model_state_dict"])
        model.load_state_dict(nnet['model_state_dict'])

    feats_config = pickle.load(open(config.egs_config, 'rb'))

    lsm = torch.nn.LogSoftmax(1)
    sm = torch.nn.Softmax(1)

    if config.override_trans:
        feat_type = config.override_trans.split(',')[0]
        trans_path = config.override_trans.split(',')[1]
    else:
        feat_type = feats_config['feat_type'].split(',')[0]
        trans_path = feats_config['feat_type'].split(',')[1]

    if feat_type == "pca":
        cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, config.scp)
    elif feat_type == "cmvn":
        cmd = "apply-cmvn --norm-vars=true {} scp:{} ark:- |".format(trans_path, config.scp)
    elif feat_type == "cmvn_utt":
        cmd = "apply-cmvn --norm-vars=true scp:{} scp:{} ark:- |".format(trans_path, config.scp)
    else:
        cmd = "copy-feats scp:{} ark:- |".format(config.scp)

    if feats_config['concat_feats']:
        context = feats_config['concat_feats'].split(',')
        cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])
    if config.prior:
        prior = pickle.load(open(config.prior, 'rb'))

    post_dict = {}
    model.eval()
    for utt_id, batch_x in kaldi_io.read_mat_ark(cmd):
        #print(batch_x.shape)
        if config.vae_arch == "cnn":
            batch_l = Variable(torch.IntTensor([batch_x.shape[0]]))
            batch_x = Variable(torch.FloatTensor(batch_x))
            batch_x = batch_x[None, None, :, :]
            batch_x = torch.transpose(batch_x, 2, 3)
            _, batch_x = vae_model(batch_x)
            batch_x = torch.transpose(batch_x[0], 1, 2)
        else:
            batch_x = Variable(torch.FloatTensor(batch_x))[None, :, :]
            batch_l = Variable(torch.IntTensor([batch_x.shape[1]]))
            _, batch_x = vae_model(batch_x, batch_l)
            batch_x = batch_x[0]
            print(batch_x.shape)

        batch_x = batch_x - torch.cat(batch_x.shape[1] * [torch.mean(batch_x, dim=1)[:, None, :]], dim=1)
        batch_x = batch_x / torch.sqrt(torch.cat(batch_x.shape[1] * [torch.var(batch_x, dim=1)[:, None, :]], dim=1))

        batch_x = model(batch_x, batch_l)

        if config.prior:
            print(batch_x[0].shape)
            sys.stdout.flush()
            post_dict[utt_id] = lsm(batch_x[0, :, :]).data.numpy() - config.prior_weight * prior
        else:
            if config.add_softmax:
                post_dict[utt_id] = sm(batch_x[0, :, :]).data.numpy()
            else:
                post_dict[utt_id] = batch_x[0, :, :].data.numpy()

    return post_dict


if __name__ == '__main__':
    config = get_args()
    post_dict = get_output(config)
    dict2Ark(post_dict, os.path.abspath(config.save_file), kaldi_cmd='copy-feats')

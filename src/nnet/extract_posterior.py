import os
import logging
import argparse
import torch
from torch.autograd import Variable
from nnet_models import nnetFeedforward
import kaldi_io
from features import dict2Ark
import numpy as np
import pickle


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def get_args():
    parser = argparse.ArgumentParser(description="Extract Posterior from a Feedforward nnet Model")

    parser.add_argument("model", help="Feedforward pytorch nnet model")
    parser.add_argument("scp", help="scp files for features")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("save_file", help="file to save posteriors")
    parser.add_argument("--layer", type=int, default=0, help="layer from the end to get the posteriors from")
    parser.add_argument("--add_softmax", action="store_true", help="Add softmax to layer zero posteriors")
    return parser.parse_args()


def load_model(config):
    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    model = nnetFeedforward(nnet['feature_dim'] * nnet['num_frames'], nnet['num_layers'], nnet['hidden_dim'],
                            nnet['num_classes'])
    model.load_state_dict(nnet['model_state_dict'])
    model.eval()

    return model


def get_post(config, nnet):
    feats_config = pickle.load(open(config.egs_config, 'rb'))

    if feats_config['feat_type']:
        feat_type = feats_config['feat_type'].split(',')[0]
        trans_path = feats_config['feat_type'].split(',')[1]

    if feat_type == "pca":
        cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, config.scp)
    elif feat_type == "cmvn":
        cmd = "apply-cmvn {} scp:{} ark:- |".format(trans_path, config.scp)
    else:
        cmd = config.scp

    if feats_config['concat_feats']:
        context = feats_config['concat_feats'].split(',')
        cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])

    post_dict = {}
    for utt_id, mat in kaldi_io.read_mat_ark(cmd):
        out = nnet(Variable(torch.FloatTensor(mat)))
        if config.layer == 0:
            if config.add_softmax:
                post_dict[utt_id] = softmax(out[1].data.numpy())
            else:
                post_dict[utt_id] = out[1].data.numpy()
        else:
            post_dict[utt_id] = out[0][-config.layer].data.numpy()

    return post_dict


if __name__ == '__main__':
    config = get_args()
    nnet = load_model(config)
    post_dict = get_post(config, nnet)
    dict2Ark(post_dict, os.path.abspath(config.save_file), kaldi_cmd='copy-feats')

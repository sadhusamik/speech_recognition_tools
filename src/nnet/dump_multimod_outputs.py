import os
import logging
import argparse
import torch
from torch.autograd import Variable
from nnet_models import nnetRNNMultimod
import kaldi_io
from features import dict2Ark
import numpy as np
import pickle
import sys


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def get_args():
    parser = argparse.ArgumentParser(description="Dump likelihoods or posteriors from a genclassifier model")

    parser.add_argument("model", help="genclassifier pytorch nnet model")
    parser.add_argument("scps", help="scp for all stream features")
    parser.add_argument("egs_configs", help="config files for generating examples")
    parser.add_argument("save_file", help="file to save posteriors")
    parser.add_argument("--prior", default=None, help="Provide prior to normalize and get likelihoods")
    parser.add_argument("--prior_weight", type=float, default=0.8, help="Weight for the prior distribution")
    parser.add_argument("--add_softmax", action="store_true", help="Set to add softmax when dumping posteriors")
    parser.add_argument("--override_trans", default=None, help="Provide a different feature transformation file")
    return parser.parse_args()


def get_output(config):
    # Load model
    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)

    model = nnetRNNMultimod(nnet['feature_dim'] * nnet['num_frames'],
                            nnet['num_layers_subband'],
                            nnet['num_layers'],
                            nnet['hidden_dim_subband'],
                            nnet['num_classes'],
                            nnet['mod_num'])

    model.load_state_dict(nnet['model_state_dict'])
    lsm = torch.nn.LogSoftmax(1)
    sm = torch.nn.Softmax(1)
    if config.prior:
        prior = pickle.load(open(config.prior, 'rb'))

    # Load all feature config
    all_feats_config = []
    egs_configs = config.egs_configs.split(',')
    for egs_config in egs_configs:
        all_feats_config.append(pickle.load(open(egs_config, 'rb')))
    if len(all_feats_config) != nnet['mod_num']:
        print("Modulation numbers not matching number of egs.config files given!")
        sys.exit(1)
    all_scps = config.scps.split(',')
    if len(all_scps) != nnet['mod_num']:
        print("Number of modulations does not match with number of scp files given!")
        sys.exit(1)

    all_feats = []
    for idx, scp in enumerate(all_scps):
        feat_type = all_feats_config[idx]['feat_type'].split(',')[0]
        trans_path = all_feats_config[idx]['feat_type'].split(',')[1]
        context = all_feats_config[idx]['concat_feats'].split(',')

        if feat_type == "pca":
            cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, scp)
        elif feat_type == "cmvn":
            cmd = "apply-cmvn --norm-vars=true {} scp:{} ark:- |".format(trans_path, scp)
        elif feat_type == "cmvn_utt":
            cmd = "apply-cmvn --norm-vars=true scp:{} scp:{} ark:- |".format(trans_path, scp)
        else:
            cmd = "copy-feats scp:{} ark:- |".format(config.scp)

        cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])
        feat_dict = {utt_id: Variable(torch.FloatTensor(mat)) for utt_id, mat in kaldi_io.read_mat_ark(cmd)}
        all_feats.append(feat_dict)

    post_dict = {}
    # I assume 3 streams from here
    for utt_id in all_feats[0]:
        batch_x1 = all_feats[0][utt_id][None, :, :]
        batch_x2 = all_feats[1][utt_id][None, :, :]
        batch_x3 = all_feats[2][utt_id][None, :, :]
        batch_l = Variable(torch.IntTensor([batch_x1.shape[1]]))
        out = model([batch_x1, batch_x2, batch_x3], batch_l)
        if config.prior:
            post_dict[utt_id] = lsm(out[0, :, :]).data.numpy() - config.prior_weight * prior
        else:
            if config.add_softmax:
                post_dict[utt_id] = sm(out[0, :, :]).data.numpy()
            else:
                post_dict[utt_id] = out[0, :, :].data.numpy()

    return post_dict


if __name__ == '__main__':
    config = get_args()
    post_dict = get_output(config)
    dict2Ark(post_dict, os.path.abspath(config.save_file), kaldi_cmd='copy-feats')

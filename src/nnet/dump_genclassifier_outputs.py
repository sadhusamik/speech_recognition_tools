import os
import logging
import argparse
import torch
from torch.autograd import Variable
from nnet_models import nnetAEClassifierMultitask, nnetVAEClassifier, nnetRNN, VAEEncodedClassifier, nnetVAE
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
    parser.add_argument("scp", help="scp files for features")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("save_file", help="file to save posteriors")
    parser.add_argument("--ae_type", default="normal", help="Type of autoencoder vae/normal/noae/vaeenc")
    parser.add_argument("--vae", default=None, help="Path to VAE for ")
    parser.add_argument("--prior", default=None, help="Provide prior to normalize and get likelihoods")
    parser.add_argument("--prior_weight", type=float, default=0.8, help="Weight for the prior distribution")
    parser.add_argument("--add_softmax", action="store_true", help="Set to add softmax when dumping posteriors")
    parser.add_argument("--override_trans", default=None, help="Provide a different feature transformation file")
    return parser.parse_args()


def get_output(config):
    # Load model
    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    if config.ae_type == "normal":
        model = nnetAEClassifierMultitask(nnet['feature_dim'] * nnet['num_frames'], nnet['num_classes'],
                                          nnet['encoder_num_layers'], nnet['classifier_num_layers'],
                                          nnet['ae_num_layers'],
                                          nnet['hidden_dim'],
                                          nnet['bn_dim'], nnet['enc_dropout'])
    elif config.ae_type == "vae":
        model = nnetVAEClassifier(nnet['feature_dim'] * nnet['num_frames'], nnet['num_classes'],
                                  nnet['encoder_num_layers'], nnet['classifier_num_layers'],
                                  nnet['ae_num_layers'],
                                  nnet['hidden_dim'],
                                  nnet['bn_dim'], nnet['enc_dropout'], use_gpu=False)
    elif config.ae_type == "noae":
        model = nnetRNN(nnet['feature_dim'] * nnet['num_frames'],
                        nnet['num_layers'],
                        nnet['hidden_dim'],
                        nnet['num_classes'], nnet['dropout'])
    elif config.ae_type == "vaeenc":
        nnet['vaeenc'] = "exp_hybrid/hybrid_lll/nnet_vae_enc2l_dec2l_300nodes/exp_1.dir/exp_1__epoch_160.model"
        vae = torch.load(nnet['vaeenc'], map_location=lambda storage, loc: storage)
        vae_model = nnetVAE(vae['feature_dim'] * vae['num_frames'], vae['encoder_num_layers'],
                            vae['decoder_num_layers'], vae['hidden_dim'], vae['bn_dim'], 0, False)
        model = VAEEncodedClassifier(vae_model,
                                     vae['bn_dim'],
                                     nnet['num_layers'],
                                     nnet['hidden_dim'], nnet['num_classes'])
    else:
        print("Model type {} not supported!".format(config.ae_type))
        sys.exit(1)

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
    for utt_id, mat in kaldi_io.read_mat_ark(cmd):
        mat = Variable(torch.FloatTensor(mat))[None, :, :]
        batch_l = Variable(torch.IntTensor([mat.size(1)]))
        if config.ae_type == "normal":
            out, _ = model(mat, batch_l)
        elif config.ae_type == "vae":
            out, _, _ = model(mat, batch_l)
        elif config.ae_type == "noae":
            out = model(mat, batch_l)
        elif config.ae_type == "vaeenc":
            out = model(mat, batch_l)

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

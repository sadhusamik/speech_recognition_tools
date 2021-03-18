import argparse
import subprocess
import os
import pickle as pkl
import numpy as np
import kaldi_io
from os import listdir
import os
import logging
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from nnet_models import nnetFeedforward
import subprocess
import pickle
import kaldi_io
from ae_model import autoencoderRNN
import sys


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def get_args():
    parser = argparse.ArgumentParser("Compute per utterance FER")
    parser.add_argument("model", help="Feedforward pytorch nnet model")
    parser.add_argument("scp", help="scp file to update model")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("ali_dir", help="Kaldi directory with alignments .gz files")
    parser.add_argument("save_fer", help="Location to save the FER dictionary")

    parser.add_argument("--ali_type", default="phone", help="phone/pdf to get phone or pdf alignment labels")
    parser.add_argument("--override_trans_path", default=None,
                        help="Override the feature transformation file used in egs config")

    return parser.parse_args()


def load_posteriors(post_scp, use_softmax=True):
    if use_softmax:
        d = {key: softmax(mat) for key, mat in kaldi_io.read_mat_scp(post_scp)}
    else:
        d = {key: mat for key, mat in kaldi_io.read_mat_scp(post_scp)}
    return d


def run(config):
    # Load the nnet model
    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    model = nnetFeedforward(nnet['feature_dim'] * nnet['num_frames'], nnet['num_layers'], nnet['hidden_dim'],
                            nnet['num_classes'])
    model.load_state_dict(nnet['model_state_dict'])

    # Load alignment
    ali_files = [os.path.join(config.ali_dir, f) for f in listdir(config.ali_dir) if f.startswith('ali.')]
    pdf_ali_dict = {}

    for file in ali_files:
        if config.ali_type == "pdf":
            pdf_ali_file = "ark:ali-to-pdf {} ark:'gunzip -c {} |' ark:- |".format(
                os.path.join(config.ali_dir, "final.mdl"),
                file)
        else:
            pdf_ali_file = "ark:ali-to-phones --per-frame {} ark:'gunzip -c {} |' ark:- |".format(
                os.path.join(config.ali_dir, "final.mdl"),
                file)
        pdf_ali_dict.update({u: d - 1 for u, d in kaldi_io.read_vec_int_ark(pdf_ali_file)})

    # Load feature stuff
    feats_config = pickle.load(open(config.egs_config, 'rb'))
    if feats_config['feat_type']:
        feat_type = feats_config['feat_type'].split(',')[0]
        trans_path = feats_config['feat_type'].split(',')[1]

    if config.override_trans_path is not None:
        trans_path = config.override_trans_path

    if feat_type == "pca":
        cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, config.scp)
    elif feat_type == "cmvn":
        cmd = "apply-cmvn {} scp:{} ark:- |".format(trans_path, config.scp)
    else:
        cmd = config.scp

    if feats_config['concat_feats']:
        context = feats_config['concat_feats'].split(',')
        cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])

    # Get the posterior
    fer_dict = {}
    for utt_id, mat in kaldi_io.read_mat_ark(cmd):
        out = model(Variable(torch.FloatTensor(mat)))
        out = softmax(out[1].data.numpy())
        als = pdf_ali_dict[utt_id]
        preds = np.argmax(out, axis=1)
        err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, als)))) * 100 / float(preds.shape[0])
        fer_dict[utt_id] = err

    return fer_dict


if __name__ == "__main__":
    config = get_args()
    fer_dict = run(config)

    with open(config.save_fer, 'wb') as f:
        pkl.dump(fer_dict, f)

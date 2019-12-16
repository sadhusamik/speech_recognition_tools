import os
import logging
import argparse
import torch
from torch.autograd import Variable
from nnet_models import nnetFeedforward
import kaldi_io
from features import dict2Ark
from os import listdir
import pickle
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Extract Posterior from a Feedforward nnet Model")

    parser.add_argument("model", help="Feedforward pytorch nnet model")
    parser.add_argument("scp", help="scp files for features")
    parser.add_argument("ali_dir", help="Kaldi alignment directory")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("save_file", help="file to save posteriors")

    parser.add_argument("--ali_type", default="phone", help="Alignment type, pdf or phone")

    return parser.parse_args()


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def get_labels(ali_dir, ali_type):
    ali_files = [os.path.join(ali_dir, f) for f in listdir(ali_dir) if f.startswith('ali.')]
    pdf_ali_dict = {}

    for file in ali_files:
        if ali_type == "pdf":
            pdf_ali_file = "ark:ali-to-pdf {} ark:'gunzip -c {} |' ark:- |".format(
                os.path.join(ali_dir, "final.mdl"),
                file)
        else:
            pdf_ali_file = "ark:ali-to-phones --per-frame {} ark:'gunzip -c {} |' ark:- |".format(
                os.path.join(ali_dir, "final.mdl"),
                file)
        pdf_ali_dict.update({u: d - 1 for u, d in kaldi_io.read_vec_int_ark(pdf_ali_file)})

    return pdf_ali_dict


def load_model(config):
    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    model = nnetFeedforward(nnet['feature_dim'] * nnet['num_frames'], nnet['num_layers'], nnet['hidden_dim'],
                            nnet['num_classes'])
    model.load_state_dict(nnet['model_state_dict'])
    model.eval()

    return model


def get_fer(config, nnet, alis):
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

    fer_dict = {}

    for utt_id, mat in kaldi_io.read_mat_ark(cmd):
        out = nnet(Variable(torch.FloatTensor(mat)))
        als = alis[utt_id]
        preds = np.argmax(softmax(out[1].data.numpy()), axis=1)
        err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, als)))) * 100 / float(preds.shape[0])
        fer_dict[utt_id] = err

    return fer_dict


if __name__ == '__main__':
    config = get_args()
    nnet = load_model(config)
    alis = get_labels(config.ali_dir, config.ali_type)
    fer_dict = get_fer(config, nnet, alis)
    pickle.dump(fer_dict, open(config.save_fie, 'wb'))

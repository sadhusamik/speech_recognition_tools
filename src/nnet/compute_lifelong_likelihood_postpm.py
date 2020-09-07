import os
import logging
import argparse
import torch
from torch.autograd import Variable
from nnet_models import nnetRNN, nnetVAE
import kaldi_io
from features import dict2Ark
import numpy as np
import pickle
import sys
from torch import nn
from scipy.stats import entropy


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def vae_loss(x, ae_out, latent_out):
    log_lhood = torch.mean(-0.5 * torch.pow((x - ae_out), 2) - 0.5 * np.log(2 * np.pi * 1), dim=1)
    kl_loss = 0.5 * torch.mean(
        1 - torch.pow(latent_out[0], 2) - torch.pow(torch.exp(latent_out[1]), 2) + 2 * latent_out[1], dim=1)
    return log_lhood + kl_loss


def sym_kld(X, Y):
    return torch.sum(X * (torch.log(X) - torch.log(Y))) / X.size(0) + torch.sum(
        Y * (torch.log(Y) - torch.log(X))) / X.size(0)


def mmeasure_loss(X, del_list=[5, 25, 45, 65], use_gpu=False):
    kld = nn.KLDivLoss()
    if use_gpu:
        m_acc = torch.FloatTensor([0]).cuda()
    else:
        m_acc = torch.FloatTensor([0])

    for d in del_list:
        m_acc += sym_kld(X[d:, :], X[0:-d, :]) + kld(X[0:-d:, :], X[d:, :])
    return m_acc / len(del_list)


def get_args():
    parser = argparse.ArgumentParser(description="Dump likelihoods or posteriors from a genclassifier model")

    parser.add_argument("models_pcx", help="All the nnet models")
    parser.add_argument("models_px", help="All the likelihood models")
    parser.add_argument("scp", help="scp files for features")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("priors",
                        help="Provide prior to normalize and get likelihoods, dp/mm/lowent prior with commas ")
    parser.add_argument("task_prior", help="Provide prior for every task")
    parser.add_argument("save_file", help="file to save posteriors")
    parser.add_argument("--prior_weight", type=float, default=0.8, help="Weight for the prior distribution")
    parser.add_argument("--override_trans", default=None, help="Provide a different feature transformation file")
    return parser.parse_args()


def get_output(config):
    # Load all P(x) and P(c|x) models
    model_pcx = config.models_pcx.split(',')
    model_px = config.models_px.split(',')

    if len(model_pcx) != len(model_px):
        print("Number of p(x) models and p(c|x) models are not the same!")
    num_domains = len(model_px)

    all_pcx_models = []
    all_px_models = []

    for idx, m in enumerate(model_pcx):
        nnet = torch.load(model_pcx[idx], map_location=lambda storage, loc: storage)
        vae = torch.load(model_px[idx], map_location=lambda storage, loc: storage)
        model = nnetRNN(nnet['feature_dim'] * nnet['num_frames'],
                        nnet['num_layers'],
                        nnet['hidden_dim'],
                        nnet['num_classes'], nnet['dropout'])
        model.load_state_dict(nnet['model_state_dict'])
        all_pcx_models.append(model)
        model = nnetVAE(nnet['num_classes'], vae['encoder_num_layers'],
                        vae['decoder_num_layers'], vae['hidden_dim'], vae['bn_dim'], 0, False)
        model.load_state_dict(vae['model_state_dict'])
        all_px_models.append(model)
        num_classes = nnet['num_classes']

    feats_config = pickle.load(open(config.egs_config, 'rb'))
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

    # Load prior
    priors = config.priors.split(',')
    priors = [pickle.load(open(f, 'rb')) for f in priors]

    if config.task_prior == "mm":
        print("using mm-measure based task priors")
    elif config.task_prior == "dp":
        print("using data based task priors")
    elif config.task_prior == "lowent":
        print("Using low-entropy prior")
    else:
        task_prior = config.task_prior.split(',')
        task_prior = [float(tp) for tp in task_prior]

    post_dict = {}
    for utt_id, mat in kaldi_io.read_mat_ark(cmd):
        post = np.zeros((mat.shape[0], num_classes))
        prior_acc = np.zeros(num_classes)
        mat = Variable(torch.FloatTensor(mat))[None, :, :]
        batch_l = Variable(torch.IntTensor([mat.size(1)]))
        px_save = []
        all_pcx = []
        all_px = []
        all_tp = []
        all_tp_2 = []
        for idx, model in enumerate(all_pcx_models):
            model.eval()
            out = model(mat, batch_l)
            ae_out, latent_out = all_px_models[idx](out, batch_l)
            latent_out = (latent_out[0][0, :, :], latent_out[1][0, :, :])
            px = np.exp(vae_loss(out[0, :, :], ae_out[0, :, :], latent_out).data.numpy())
            px_save.append(np.mean(px))
            pcx = sm(out[0, :, :])
            px = np.tile(px, (pcx.shape[1], 1)).T
            all_pcx.append(pcx.data.numpy())
            all_px.append(np.ones(px.shape))

            if config.task_prior == "mm":
                mm = mmeasure_loss(pcx).item()
                all_tp.append(mm)
                print("task {:d} , mm={:.2f}".format(idx, mm))
            elif config.task_prior == "dp":
                all_tp.append(px_save[idx])
            elif config.task_prior == "lowent":
                mm = mmeasure_loss(pcx).item()
                all_tp.append(mm)
                all_tp_2.append(px_save[idx])
            else:
                all_tp.append(task_prior[idx])

        if config.task_prior == "mm":
            all_tp = np.asarray(all_tp, dtype=np.float64)
            all_tp = np.exp(all_tp) / np.sum(np.exp(all_tp))
            if np.isnan(all_tp[0]):
                print("Switching to uniform priors")
                all_tp = np.ones(num_domains) / num_domains
        elif config.task_prior == "dp":
            all_tp = np.asarray(all_tp, dtype=np.float64)
            all_tp = np.exp(300 * all_tp) / np.sum(np.exp(300 * all_tp))
        elif config.task_prior == "lowent":
            all_tp = np.asarray(all_tp)
            all_tp = np.exp(all_tp) / np.sum(np.exp(all_tp))
            if np.isnan(all_tp[0]):
                print("Switching to uniform priors")
                all_tp = np.ones(num_domains) / num_domains
            all_tp_2 = np.asarray(all_tp_2)
            all_tp_2 = np.exp(200 * all_tp_2) / np.sum(np.exp(200 * all_tp_2))
            print('Entropy dp:{:.2f} and Entropy mm:{:.2f}'.format(entropy(all_tp_2), entropy(all_tp)))
            if entropy(all_tp_2) < entropy(all_tp):
                all_tp = all_tp_2
        for idx, pcx in enumerate(all_pcx):
            post += pcx * all_px[idx] * all_tp[idx]
            prior_acc += np.exp(priors[idx]) * all_tp[idx]

        print_log = ""
        for ii, x in enumerate(px_save):
            print_log += "p(x) for Task {:d} ={:.6f} with prior ={:.6f} ".format(ii, x, all_tp[ii])
        print(print_log)
        sys.stdout.flush()
        post_dict[utt_id] = np.log(post) - config.prior_weight * np.log(prior_acc)

    return post_dict


if __name__ == '__main__':
    config = get_args()
    post_dict = get_output(config)
    dict2Ark(post_dict, os.path.abspath(config.save_file), kaldi_cmd='copy-feats')

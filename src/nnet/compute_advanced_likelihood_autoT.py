import os
from torch import nn
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
import subprocess


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


def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]


def powerset(A):
    if A == []:
        return [[]]
    a = A[0]
    incomplete_pset = powerset(A[1:])
    rest = []
    for set in incomplete_pset:
        rest.append([a] + set)
    return rest + incomplete_pset


def get_args():
    parser = argparse.ArgumentParser(description="Compute advanced lifelong likelihoods")

    parser.add_argument("models_pcx", help="All the nnet models")
    parser.add_argument("models_px", help="All the likelihood models")
    parser.add_argument("scp", help="scp files for features")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("priors",
                        help="Provide prior to normalize and get likelihoods, dp/mm/lowent prior with commas ")
    parser.add_argument("task_prior", help="Provide prior for every task")
    parser.add_argument("save_file", help="file to save posteriors")
    parser.add_argument("--lr_rate", type=float, default=10,
                        help="Learning rate for max likelihood gradient descent")
    parser.add_argument("--num_iter", type=int, default=20, help="Number of iteration for max likeihood training")
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--prior_weight", type=float, default=0.8, help="Weight for the prior distribution")
    parser.add_argument("--override_trans", default=None, help="Provide a different feature transformation file")
    return parser.parse_args()


def compute_lhood(num_frames, num_classes, all_pcx, all_tp, priors, task_prior, streams, temperature):
    num = torch.ones(num_frames, num_classes, dtype=torch.double)
    denom = torch.ones(num_classes, dtype=torch.double)
    if task_prior == "dp":
        all_tp = torch.exp(temperature * all_tp) / torch.sum(torch.exp(temperature * all_tp))

    print(all_tp)
    for idx, st in enumerate(streams):
        num_prod = torch.ones(num_frames, num_classes, dtype=torch.double)
        denom_prod = torch.ones(num_classes, dtype=torch.double)
        perf_mon = torch.DoubleTensor([1])

        for b in st:
            num_prod *= torch.pow(all_pcx[b], all_tp[b])
            perf_mon *= all_tp[b]
            denom_prod *= torch.exp(priors[b])

        denom_prod /= torch.sum(denom_prod)
        num_prod = num_prod / torch.sum(num_prod, dim=1).view(-1, 1).repeat(1, num_prod.shape[1])
        num += num_prod * perf_mon
        denom += denom_prod
    return torch.log(num) - config.prior_weight * torch.log(denom)


def get_output(config):
    # Load all P(x) and P(c|x) models
    model_pcx = config.models_pcx.split(',')
    model_px = config.models_px.split(',')

    if len(model_pcx) != len(model_px):
        print("Number of p(x) models and p(c|x) models are not the same!")
    num_domains = len(model_px)
    streams = powerset(list(np.arange(num_domains)))

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
        model = nnetVAE(vae['feature_dim'] * vae['num_frames'], vae['encoder_num_layers'],
                        vae['decoder_num_layers'], vae['hidden_dim'], vae['bn_dim'], 0, False)
        model.load_state_dict(vae['model_state_dict'])
        all_px_models.append(model)
        num_classes = nnet['num_classes']

    feats_config = pickle.load(open(config.egs_config, 'rb'))

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

    if config.use_gpu:
        priors = [torch.from_numpy(f).cuda().double() for f in priors]
    else:
        priors = [torch.from_numpy(f).double() for f in priors]

    all_pcx_models = nn.ModuleList(all_pcx_models)
    all_px_models = nn.ModuleList(all_px_models)

    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id
        all_pcx_models.cuda()
        all_px_models.cuda()

    if config.task_prior == "dp":
        print("using data based task priors")
        task_prior = "dp"
    else:
        task_prior = config.task_prior.split(',')
        task_prior = [float(tp) for tp in task_prior]

    post_dict = {}
    for utt_id, batch_x in kaldi_io.read_mat_ark(cmd):
        print("COMPUTING LOG-LIKELIHOOD FOR UTTERANCE {:s}".format(utt_id))
        sys.stdout.flush()

        T = torch.DoubleTensor([300])  # Initial temperature
        T.requires_grad = True
        num_frames = batch_x.shape[0]

        batch_x = Variable(torch.FloatTensor(batch_x))[None, :, :]
        batch_l = Variable(torch.IntTensor([batch_x.size(1)]))

        # Do forward passes through different models

        sm = torch.nn.Softmax(1)
        px_save = []
        all_pcx = []
        all_tp = torch.zeros(len(all_pcx_models), dtype=torch.double)
        for idx, model in enumerate(all_pcx_models):
            model.eval()
            out = model(batch_x, batch_l)
            ae_out, latent_out = all_px_models[idx](batch_x, batch_l)
            latent_out = (latent_out[0][0, :, :], latent_out[1][0, :, :])
            px = torch.exp(vae_loss(batch_x[0, :, :], ae_out[0, :, :], latent_out)).double()
            px_save.append(torch.mean(px))
            pcx = sm(out[0, :, :])
            all_pcx.append(pcx.double())

            if task_prior == "dp":
                all_tp[idx] = px_save[idx]
            else:
                all_tp[idx] = task_prior[idx]

        for it_num in range(config.num_iter):
            llh = compute_lhood(num_frames, num_classes, all_pcx, all_tp, priors, task_prior, streams, T)

            loss = -torch.mean(llh)
            print_log = "p(x|c) ={:.6f} with softmax temperature ={:.6f} ".format(loss.item(), T.item())
            print(print_log)
            sys.stdout.flush()
            #loss.backward(retain_graph=True)
            print(T.grad)
            # with torch.no_grad():
            # T = T + config.lr_rate * T.grad/torch.norm(T.grad,2)
            #T.requires_grad = True
            T = T + 100
        if config.use_gpu:
            post_dict[utt_id] = llh.cpu().data.numpy()
        else:
            post_dict[utt_id] = llh.data.numpy()

    return post_dict


if __name__ == '__main__':
    config = get_args()
    post_dict = get_output(config)
    dict2Ark(post_dict, os.path.abspath(config.save_file), kaldi_cmd='copy-feats')

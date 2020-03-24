import os
import logging
import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from nnet_models import nnetFeedforward
import subprocess
import pickle
import kaldi_io
from ae_model import seq2seqRNNAE
import sys


def get_args():
    parser = argparse.ArgumentParser(
        description="Get PM scores for each utterance with seq 2 seq RNN-AE")

    parser.add_argument("model", help="Feedforward pytorch nnet model")
    parser.add_argument("pms", help="List of RNN AE performance monitoring model for different layers")
    parser.add_argument("scp", help="scp file to update model")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("cmvns", help="All global cmvn files for output from different layers")

    # Other options
    parser.add_argument("--override_trans_path", default=None,
                        help="Override the feature transformation file used in egs config")
    parser.add_argument("--pm_index", default="0,-1,-2,-3,-4",
                        help="Index of pms passed as number of layers from output layer")
    parser.add_argument("--decoder_input", action="store_true",
                        help="Set to use the input sequence as input to decoder apart from encoder state")
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--log_file", default="log.log", type=str, help="Print Log file")
    parser.add_argument("--out_file", default="pm.score", type=str, help="Output scoring file")
    parser.add_argument("--loss", default="MSE", help="Loss function L1/MSE")

    return parser.parse_args()


def samplewise_mse(x, y):
    return torch.mean(((x - y) ** 2), dim=2)


def samplewise_abs(x, y):
    return torch.mean(torch.abs(x - y), dim=2)


def stable_mse(x, y):
    return ((x - y) ** 2).mean()


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]


def adjust_learning_rate(optimizer, lr, f):
    lr = lr * f
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr


def get_cmvn(cmvn_file):
    shell_cmd = "copy-matrix --binary=false {:s} - ".format(cmvn_file)
    r = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE)
    r = r.stdout.decode('utf-8').split('\n')

    r_m = r[1].strip().split()
    r_v = r[2].strip().split()
    frame_num = float(r_m[-1])
    means = np.asarray([float(x) / frame_num for x in r_m[0:-1]])
    var = np.asarray([float(x) / frame_num for x in r_v[0:-2]])

    return means, var


def update(config):
    # Load model

    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    model = nnetFeedforward(nnet['feature_dim'] * nnet['num_frames'], nnet['num_layers'], nnet['hidden_dim'],
                            nnet['num_classes'])
    model.load_state_dict(nnet['model_state_dict'])

    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id

        model = model.cuda()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        filename=config.log_file,
        filemode='w')

    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Model Parameters: ')
    logging.info('Number of Layers: %d' % (nnet['num_layers']))
    logging.info('Hidden Dimension: %d' % (nnet['feature_dim']))
    logging.info('Number of Classes: %d' % (nnet['num_classes']))
    logging.info('Data dimension: %d' % (nnet['feature_dim']))
    logging.info('Number of Frames: %d' % (nnet['num_frames']))

    if config.loss == "MSE":
        criterion = samplewise_mse
    elif config.loss == "L1":
        criterion = samplewise_abs
    else:
        logging.info('Loss function {:s} is not supported'.format(config.loss))
        sys.exit(1)

    pi = [int(t) for t in config.pm_index.split(',')]

    # Figure out all feature stuff
    shuff_file = config.scp
    feats_config = pickle.load(open(config.egs_config, 'rb'))

    if feats_config['feat_type']:
        feat_type = feats_config['feat_type'].split(',')[0]
        trans_path = feats_config['feat_type'].split(',')[1]

    if config.override_trans_path is not None:
        trans_path = config.override_trans_path

    if feat_type == "pca":
        cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, shuff_file)
    elif feat_type == "cmvn":
        cmd = "apply-cmvn {} scp:{} ark:- |".format(trans_path, shuff_file)
    else:
        cmd = shuff_file

    if feats_config['concat_feats']:
        context = feats_config['concat_feats'].split(',')
        cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])

    # Load performance monitoring models
    pm_paths = config.pms.split(',')

    pm_models = []
    feat_dims = []
    for path in pm_paths:

        pm_model = torch.load(path, map_location=lambda storage, loc: storage)
        ae_model = seq2seqRNNAE(pm_model['feature_dim'], pm_model['feature_dim'],
                                pm_model['encoder_num_layers'], pm_model['decoder_num_layers'],
                                pm_model['hidden_dim'], False, config.decoder_input)
        ae_model.load_state_dict(pm_model['model_state_dict'])
        feat_dims.append(pm_model['feature_dim'])

        if config.use_gpu:
            ae_model.cuda()

        for p in ae_model.parameters():  # Do not update performance monitoring block
            p.requires_grad = False

        pm_models.append(ae_model)

    pm_paths = config.pms.split(',')
    if len(pi) != len(pm_paths):
        logging.error("Number of pm models {:d} and number indices {:d} do not match".format(len(pm_paths), len(pi)))
        sys.exit(0)

    cmvn_paths = config.cmvns.split(',')
    means = []
    for path in cmvn_paths:
        mean, _ = get_cmvn(path)
        means.append(mean)

    if len(cmvn_paths) != len(pm_paths):
        logging.error("Number of cmvn paths not equal to number of model paths, exiting training!")
        sys.exit(1)
    else:
        num_pm_models = len(pm_paths)

    pm_scores = {}
    for idx in range(num_pm_models):
        pm_scores[idx] = {}

    for utt_id, mat in kaldi_io.read_mat_ark(cmd):
        batches = []
        lens = mat.shape[0]

        if config.use_gpu:
            out = model(Variable(torch.FloatTensor(mat)).cuda())
        else:
            out = model(Variable(torch.FloatTensor(mat)))

        for idx in range(num_pm_models):
            if config.use_gpu:
                if pi[idx] == 0:
                    post = out[1] - Variable(torch.FloatTensor(means[idx])).cuda()
                else:
                    post = out[0][pi[idx]] - Variable(torch.FloatTensor(means[idx])).cuda()

            else:
                if pi[idx] == 0:
                    post = out[1] - Variable(torch.FloatTensor(means[0]))
                else:
                    post = out[0][pi[idx]] - Variable(torch.FloatTensor(means[idx]))

            batches.append(post)

        ## Get the PM scores
        lens = torch.IntTensor([lens])

        for idx in range(num_pm_models):
            batch_x = batches[idx]
            batch_x = batch_x[None, :, :]
            ae_model = pm_models[idx]
            batch_l = lens

            outputs = ae_model(batch_x, batch_l)
            loss = criterion(outputs, batch_x).mean()
            pk = pm_scores[idx]
            pk[utt_id] = loss.item()
            pm_scores[idx] = pk

    pickle.dump(pm_scores, open(os.path.join(config.out_file), "wb"))


if __name__ == '__main__':
    config = get_args()
    update(config)

import os
import logging
import argparse
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
from nnet_models import nnetAEClassifierMultitaskAEAR
import pickle as pkl
from os.path import join
import subprocess


def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def compute_fer(x, l):
    x = softmax(x)
    preds = np.argmax(x, axis=1)
    err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, l)))) * 100 / float(preds.shape[0])
    return err


def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


def get_args():
    parser = argparse.ArgumentParser(
        description="AEAR Model performance monitoring score")

    parser.add_argument("model", help="AEAR Model to use for PM scoring")
    parser.add_argument("egs_dir", type=str, help="Path to the preprocessed data")
    parser.add_argument("score_path", type=str, help="File to save the pm scores")

    parser.add_argument("--pm_set", default="dev", help="Name of pm dataset")
    parser.add_argument("--ae_loss", default="MSE", help="Loss function L1/MSE")
    parser.add_argument("--log_name", default="exp_run", type=str, help="Name of this experiment")

    return parser.parse_args()


def get_score(config):
    pm_dir = os.path.dirname(config.score_path)
    os.makedirs(pm_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(pm_dir, config.log_name),
        filemode='w')
    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Load model
    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    model = nnetAEClassifierMultitaskAEAR(nnet['feature_dim'] * nnet['num_frames'], nnet['num_classes'],
                                          nnet['encoder_num_layers'], nnet['classifier_num_layers'],
                                          nnet['ae_num_layers'],
                                          nnet['hidden_dim'],
                                          nnet['bn_dim'], nnet['time_shift'])
    model.load_state_dict(nnet['model_state_dict'])

    criterion_classifier = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

    # Load dataset
    path = join(config.egs_dir, config.pm_set)
    with open(join(path, 'lengths.pkl'), 'rb') as f:
        lengths = pkl.load(f)
    labels = torch.load(join(path, 'labels.pkl'))
    ids = list(labels.keys())

    scores = {}
    model.eval()
    for utt_id in ids:
        batch_x = torch.load(join(path, utt_id))
        batch_l = torch.IntTensor([lengths[utt_id]])
        lab = labels[utt_id]
        batch_x = Variable(batch_x[None, :, :])
        batch_l = Variable(batch_l)
        lab = Variable(lab)
        class_out, ae_out, ar_out = model(batch_x, batch_l)

        l_c = criterion_classifier(class_out[0, :], lab)
        l_ae = criterion_ae(ae_out[0, :], batch_x[0, :])
        l_ar = criterion_ae(ar_out[0, :], batch_x[0, nnet['time_shift']:, :])
        fer = compute_fer(class_out[0, :].data.numpy(), lab.data.numpy())
        scores[utt_id] = [l_c.item(), l_ae.item(), l_ar.item(), fer]
        print_log = "Utt-id: {:s} AE loss: {:.3f} :: AR loss: {:.3f} :: Tr FER: {:.2f}".format(utt_id, l_ae.item(),
                                                                                               l_ar.item(), fer)
        logging.info(print_log)
        sys.stdout.flush()
    pkl.dump(scores, open(config.score_path, 'wb'))


if __name__ == '__main__':
    config = get_args()
    get_score(config)

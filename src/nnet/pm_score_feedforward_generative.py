import os
import logging
import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from nnet_models import nnetAEClassifierMultitask
import subprocess
import pickle
import kaldi_io
from datasets import nnetDatasetSeq
import sys


def get_args():
    parser = argparse.ArgumentParser(
        description="Get PM scores with feedforward generative model")

    parser.add_argument("model", help="Feedforward pytorch nnet model")
    parser.add_argument("egs_dir", type=str, help="Path to the preprocessed data")

    # Other options
    parser.add_argument("--test_set", default="test", help="Test set to compute PM score")
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--log_file", default="log.log", type=str, help="Print Log file")
    parser.add_argument("--out_file", default="pm.score", type=str, help="Output scoring file")

    return parser.parse_args()


def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


def compute_fer(x, l):
    x = softmax(x)
    preds = np.argmax(x, axis=1)
    err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, l)))) * 100 / float(preds.shape[0])
    return err


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


def compute_score(config):
    # Load model
    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    model = nnetAEClassifierMultitask(nnet['feature_dim'] * nnet['num_frames'], nnet['num_classes'],
                                      nnet['encoder_num_layers'], nnet['classifier_num_layers'], nnet['ae_num_layers'],
                                      nnet['hidden_dim'],
                                      30)
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
    logging.info('Encoder Number of Layers: %d' % (nnet['encoder_num_layers']))
    logging.info('Classifier Number of Layers: %d' % (nnet['classifier_num_layers']))
    logging.info('AE Number of Layers: %d' % (nnet['ae_num_layers']))
    logging.info('Hidden Dimension: %d' % (nnet['hidden_dim']))
    logging.info('Number of Classes: %d' % (nnet['num_classes']))
    logging.info('Data dimension: %d' % (nnet['feature_dim']))
    logging.info('Number of Frames: %d' % (nnet['num_frames']))

    # Criterion
    criterion_ae = nn.MSELoss()
    criterion_classifier = nn.CrossEntropyLoss()

    # Load test data
    path = os.path.join(config.egs_dir, config.test_set)
    file_ids = [f for f in os.listdir(path) if f.endswith('.pt')]
    with open(os.path.join(path, 'lengths.pkl'), 'rb') as f:
        lengths = pickle.load(f)
    labels = torch.load(os.path.join(path, 'labels.pkl'))

    pm_scores = {}
    for utt_id in file_ids:
        batch_x = Variable(torch.load(os.path.join(path, utt_id)))[None, :, :]
        batch_l = Variable(torch.IntTensor([lengths[utt_id]]))
        lab = Variable(labels[utt_id])[None, :]

        # Main forward pass
        class_out, ae_out = model(batch_x, batch_l)

        # Convert all the weird tensors to frame-wise form
        class_out = pad2list(class_out, batch_l)
        batch_x = pad2list(batch_x, batch_l)
        ae_out = pad2list(ae_out, batch_l)
        lab = pad2list(lab, batch_l)

        loss_classifier = criterion_classifier(class_out, lab)
        loss_ae = criterion_ae(ae_out, batch_x)
        if config.use_gpu:
            loss_fer = compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy())
        else:
            loss_fer = compute_fer(class_out.data.numpy(), lab.data.numpy())

        pm_scores[utt_id[0:-3]] = [loss_ae.item(), loss_classifier.item(), loss_fer]

    pickle.dump(pm_scores, open(os.path.join(config.out_file), "wb"))


if __name__ == '__main__':
    config = get_args()
    compute_score(config)

import argparse
import numpy as np
import torch
from torch.autograd import Variable
import kaldi_io
import pickle as pkl
import sys

def get_args():
    parser = argparse.ArgumentParser('Compute m-measure scores for a set of test utterances')
    parser.add_argument('post_scp', help="scp file for the test set posteriors")
    parser.add_argument('save_score', help="Location to save the score dictionary")
    parser.add_argument('--delta_list', default='5,15,25,35,45,55,65,75',
                        help="List of frame deltas used to compute m-measure separated by comma")

    return parser.parse_args()


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def get_mmeasure_loss(feats, delta_frame, add_softmax=True):
    if add_softmax:
        feats = softmax(feats)
    frame_num, dim = feats.shape

    m_acc = 0
    for frame_now in range(delta_frame, frame_num):
        m_acc += symmetric_KL(feats[frame_now], feats[frame_now - delta_frame])

    return m_acc / (frame_num - delta_frame)


def symmetric_KL(x, y):
    return np.sum(x * np.log(x / y)) + np.sum(y * np.log(y / x))


def compute_mmeasure(feats, delta_list):
    acc = 0
    for d in delta_list:
        acc += get_mmeasure_loss(feats, d)

    return acc / len(delta_list)


def run(config):
    mm_dict = {}
    delta_list = config.delta_list.strip().split(',')
    delta_list = [int(x) for x in delta_list]
    for key, mat in kaldi_io.read_mat_scp(config.post_scp):
        print('Computing m-measure score for utterance {}'.format(key))
        sys.stdout.flush()
        mm_dict[key] = compute_mmeasure(mat, delta_list)

    return mm_dict


if __name__ == '__main__':
    config = get_args()

    mm_dict = run(config)

    with open(config.save_score, 'wb') as f:
        pkl.dump(mm_dict, f)

import argparse
import subprocess
import os
import pickle as pkl
import numpy as np
import kaldi_io
from os import listdir


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def get_args():
    parser = argparse.ArgumentParser("Compute per utterance FER form Kaldi decode folder")
    parser.add_argument("post_scp", help="posterior scp file")
    parser.add_argument("ali_dir", help="Kaldi directory with alignments .gz files")
    parser.add_argument("save_fer", help="Location to save the FER dictionary")

    return parser.parse_args()


def load_posteriors(post_scp, use_softmax=True):
    if use_softmax:
        d = {key: softmax(mat) for key, mat in kaldi_io.read_mat_scp(post_scp)}
    else:
        d = {key: mat for key, mat in kaldi_io.read_mat_scp(post_scp)}
    return d


def run(config, post_dict):
    ali_files = [os.path.join(config.ali_dir, f) for f in listdir(config.ali_dir) if f.endswith('.gz')]
    fer_dict = {}
    for file in ali_files:
        pdf_ali_file = "ark:ali-to-pdf {} ark:'gunzip -c {} |' ark:- |".format(
            os.path.join(config.ali_dir, "final.mdl"),
            file)
        pdf_ali_dict = {u: d for u, d in kaldi_io.read_vec_int_ark(pdf_ali_file)}

        for key in pdf_ali_dict:
            als = pdf_ali_dict[key]
            preds = np.argmax(post_dict[key], axis=1)
            err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, als)))) * 100 / float(preds.shape[0])
            fer_dict[key] = err

    return fer_dict


if __name__ == "__main__":
    config = get_args()
    fer_dict = run(config, load_posteriors(config.post_scp))

    with open(config.save_fer, 'wb') as f:
        pkl.dump(fer_dict, f)

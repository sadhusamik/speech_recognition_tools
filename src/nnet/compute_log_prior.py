import os
import argparse
import kaldi_io
from os import listdir
import pickle as pkl
import numpy as np


def get_config():
    parser = argparse.ArgumentParser("Compute class prior from Kaldi alignment files")
    parser.add_argument("ali_dir", type=str,
                        help="Kaldi alignment directory with ali.gz files")
    parser.add_argument("save_file", type=str,
                        help="File to save prior file")
    parser.add_argument("--ali_type", default="phone", help="phone/pdf to get phone or pdf alignment labels")
    parser.add_argument("--num_classes", default=38, type=int, help="Number of classes")
    return parser.parse_args()


def compute_prior(config):
    ali_files = [os.path.join(config.ali_dir, f) for f in listdir(config.ali_dir) if f.startswith('ali.')]
    p = np.zeros(config.num_classes)

    for file in ali_files:
        if config.ali_type == "pdf":
            pdf_ali_file = "ark:ali-to-pdf {} ark:'gunzip -c {} |' ark:- |".format(
                os.path.join(config.ali_dir, "final.mdl"),
                file)
            for key, ali in kaldi_io.read_vec_int_ark(pdf_ali_file):
                for x in range(config.num_classes):
                    p[x] += len(np.where((ali) == x)[0])
        else:
            pdf_ali_file = "ark:ali-to-phones --per-frame {} ark:'gunzip -c {} |' ark:- |".format(
                os.path.join(config.ali_dir, "final.mdl"),
                file)
            for key, ali in kaldi_io.read_vec_int_ark(pdf_ali_file):
                for x in range(config.num_classes):
                    p[x] += len(np.where((ali-1) == x)[0])

    return np.log(p / np.sum(p))


if __name__ == '__main__':
    config = get_config()
    p = compute_prior(config)

    pkl.dump(p, open(config.save_file, 'wb'))

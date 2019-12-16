import os
import argparse
from os import listdir
import torch
import kaldi_io
import subprocess
import numpy as np
import sys
import pickle as pkl


def get_config():
    parser = argparse.ArgumentParser("Prepare Data for Feedforward AM Model Training")
    parser.add_argument("feat_file_scp", type=str,
                        help="Feature file scp")
    parser.add_argument("ali_dir", type=str,
                        help="Kaldi alignment directory with ali.gz files")
    parser.add_argument("save_dir", type=str,
                        help="Directory to save all the processed data")
    parser.add_argument("--ali_type", default="phone", help="phone/pdf to get phone or pdf alignment labels")
    parser.add_argument("--uttwise", action="store_true", help="Set flag to save utterance-wise .pt files")
    parser.add_argument("--num_chunks", default=2, type=int, help="Number of chunks to divide the data")
    parser.add_argument("--feat_type", type=str, default=None,
                        help="feat_type(cmvn/pca),path_to_cmvn or pca mat, set as None for raw features")
    parser.add_argument("--concat_feats", default=None, help="Put left and right context as left,right")
    parser.add_argument("--chunk_size", type=int, default=5, help="Size of each chunk of ark files for data processing")

    return parser.parse_args()


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


def split_scp(scp, num_chunks):
    dir = os.path.dirname(scp)

    # shuffle scp file
    scp_shuffled = os.path.join(dir, "scp_shuffled")
    cmd = "cat {} | shuf > {}".format(scp, scp_shuffled)
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)

    scps = ""
    for x in range(0, num_chunks):
        scps += " {}".format(os.path.join(dir, 'scp_split' + '.' + str(x)))

    cmd = "utils/split_scp.pl {} {}".format(scp_shuffled, scps)
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)

    return scps.split()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def dump_uttwise_feats(scp, config):
    if config.feat_type:
        feat_type = config.feat_type.split(',')[0]
        trans_path = config.feat_type.split(',')[1]

    if feat_type == "pca":
        cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, scp)
    elif feat_type == "cmvn":
        cmd = "apply-cmvn {} scp:{} ark:- |".format(trans_path, scp)
    else:
        cmd = scp

    if config.concat_feats:
        context = config.concat_feats.split(',')
        cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])

    for utt_id, mat in kaldi_io.read_mat_ark(cmd):
        one_feat = torch.FloatTensor(mat)
        torch.save(one_feat, os.path.join(config.save_dir, utt_id + '.pt'))


def dump_chunkwise_feats(scps, labels, config):
    if config.feat_type:
        feat_type = config.feat_type.split(',')[0]
        trans_path = config.feat_type.split(',')[1]
    print(scps)
    sys.stdout.flush()

    for idx, scp in enumerate(scps):
        print(scp)
        sys.stdout.flush()
        if feat_type == "pca":
            cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, scp)
        elif feat_type == "cmvn":
            cmd = "apply-cmvn {} scp:{} ark:- |".format(trans_path, scp)
        else:
            cmd = scp

        if config.concat_feats:
            context = config.concat_feats.split(',')
            cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])

        feats = {key: mat for key, mat in kaldi_io.read_mat_ark(cmd)}
        feats = [np.hstack((feats[x], labels[x][:, np.newaxis])) for x in list(feats.keys()) if x in labels]
        feats = torch.FloatTensor(np.vstack(feats))
        torch.save(feats, os.path.join(config.save_dir, "chunk_{}.pt".format(str(idx))))


def run():
    config = get_config()
    if config.uttwise:
        dump_uttwise_feats(config.feat_file_scp, config)
    else:
        alis = get_labels(config.ali_dir, config.ali_type)
        scps = split_scp(config.feat_file_scp, config.num_chunks)
        dump_chunkwise_feats(scps, alis, config)

    egs_path = os.path.dirname(config.save_dir)
    if not os.path.isfile(os.path.join(egs_path, 'egs.config')):
        egs_config = {}
        egs_config['feat_type'] = config.feat_type
        egs_config['concat_feats'] = config.concat_feats
        pkl.dump(egs_config, open(os.path.join(egs_path, 'egs.config'), 'wb'))


if __name__ == "__main__":
    run()

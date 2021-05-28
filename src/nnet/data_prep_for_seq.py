import os
import argparse
import pickle

import torch
import torch.nn.functional as F

import kaldi_io
import multiprocessing
from os import listdir
import subprocess
import pickle as pkl


def get_config():
    parser = argparse.ArgumentParser("Prepare Data for training sequence based models in pyTorch")
    parser.add_argument("feat_file_scp", type=str,
                        help="Feature scp file")
    parser.add_argument("ali_dir", type=str,
                        help="Kaldi alignment directory with ali.gz files")
    parser.add_argument("save_dir", type=str,
                        help="Directory to save all the processed data")
    parser.add_argument("--num_jobs", type=int, default=5, help="Number of parallel jobs to run")
    parser.add_argument("--notruncpad", action="store_true", help="Set to not truncate or pad the features")
    parser.add_argument("--feat_type", type=str, default=None,
                        help="feat_type(cmvn/cmvn_utt/pca),path_to_cmvn or pca mat, set as None for raw features")
    parser.add_argument("--concat_feats", default=None, help="Put left and right context as left,right")
    parser.add_argument("--ali_type", default="phone",
                        help="phone/pdf/ignore to get phone or pdf alignment labels or ignore alignment dumping")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum length (number of frames) of each sequence; sequences will be truncated or padded (with zero vectors) to this length")

    return parser.parse_args()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_scp(scp, num_jobs):
    dir = os.path.dirname(scp)

    # Divide scp file into chunks
    scps = ""
    for x in range(0, num_jobs):
        scps += " {}".format(os.path.join(dir, 'scp_split' + '.' + str(x)))

    cmd = "utils/split_scp.pl {} {}".format(scp, scps)
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)

    return scps.split()


def get_labels(ali_dir, ali_type, config):
    ali_files = []
    all_ali_dirs = ali_dir.split(',')
    for ali_dir in all_ali_dirs:
        ali_files.extend([os.path.join(ali_dir, f) for f in listdir(ali_dir) if f.startswith('ali.')])

    pdf_ali_dict = {}

    for file in ali_files:
        if ali_type == "pdf":
            pdf_ali_file = "ark:ali-to-pdf {} ark:'gunzip -c {} |' ark:- |".format(
                os.path.join(ali_dir, "final.mdl"),
                file)
            if config.notruncpad:
                pdf_ali_dict.update(
                    {u + ".pt": torch.FloatTensor(d - 0).long() for u, d in
                     kaldi_io.read_vec_int_ark(pdf_ali_file)})
            else:
                pdf_ali_dict.update(
                    {u + ".pt": F.pad(torch.FloatTensor(d - 0).long(), (0, config.max_seq_len - d.shape[0])) for u, d in
                     kaldi_io.read_vec_int_ark(pdf_ali_file)})
        else:
            pdf_ali_file = "ark:ali-to-phones --per-frame {} ark:'gunzip -c {} |' ark:- |".format(
                os.path.join(ali_dir, "final.mdl"),
                file)
            if config.notruncpad:
                pdf_ali_dict.update(
                    {u + ".pt": torch.FloatTensor(d - 1).long() for u, d in
                     kaldi_io.read_vec_int_ark(pdf_ali_file)})
            else:
                pdf_ali_dict.update(
                    {u + ".pt": F.pad(torch.FloatTensor(d - 1).long(), (0, config.max_seq_len - d.shape[0])) for u, d in
                     kaldi_io.read_vec_int_ark(pdf_ali_file)})

    torch.save(pdf_ali_dict, os.path.join(config.save_dir, 'labels.pkl'))


def dump_uttwise_feats(scp, config):
    pid = str(os.getpid())

    id2len = {}
    if config.feat_type:
        feat_type = config.feat_type.split(',')[0]
        trans_path = config.feat_type.split(',')[1]

    if config.feat_type:
        if feat_type == "pca":
            cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, scp)
        elif feat_type == "cmvn":
            cmd = "apply-cmvn --norm-vars=true {} scp:{} ark:- |".format(trans_path, scp)
        elif feat_type == "cmvn_utt":
            cmd = "apply-cmvn --norm-vars=true scp:{} scp:{} ark:- |".format(trans_path, scp)
    else:
        cmd = "copy-feats scp:{} ark:- |".format(scp)

    if config.concat_feats:
        context = config.concat_feats.split(',')
        cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])

    feats = {key: mat for key, mat in kaldi_io.read_mat_ark(cmd)}

    for utt_id in feats:
        one_feat = feats[utt_id]
        if config.notruncpad:
            id2len[utt_id + '.pt'] = one_feat.shape[0]
        else:
            id2len[utt_id + '.pt'] = min(one_feat.shape[0], config.max_seq_len)
        one_feat = torch.FloatTensor(one_feat)  # convert the 2D list to a pytorch tensor
        if not config.notruncpad:
            one_feat = F.pad(one_feat, (0, 0, 0, config.max_seq_len - one_feat.size(0)))  # pad or truncate

        torch.save(one_feat, os.path.join(config.save_dir, utt_id + '.pt'))

    with open(os.path.join(config.save_dir, pid + '.lns'), 'wb') as f:
        pickle.dump(id2len, f)


def get_all_lengths(save_dir):
    length_dict = {}
    length_files = [os.path.join(save_dir, f) for f in listdir(save_dir) if f.endswith('.lns')]

    for x in length_files:
        with open(x, 'rb') as f:
            len_dict = pickle.load(f)
        length_dict.update(len_dict)

    with open(os.path.join(save_dir, 'lengths.pkl'), 'wb') as f:
        pickle.dump(length_dict, f)


def run():
    config = get_config()
    scps = split_scp(config.feat_file_scp, config.num_jobs)

    feat_gen_processes = []
    for x in scps:
        p = multiprocessing.Process(target=dump_uttwise_feats, args=(x, config))
        feat_gen_processes.append(p)
        p.start()

    for p in feat_gen_processes:
        p.join()

    get_all_lengths(config.save_dir)
    if config.ali_type == 'ignore':
        print('No alignment directory provided, not dumping alignment')
    else:
        get_labels(config.ali_dir, config.ali_type, config)


    # if not os.path.isfile(os.path.join(egs_path, 'egs.config')):
    egs_config = {}
    egs_config['feat_type'] = config.feat_type
    egs_config['concat_feats'] = config.concat_feats
    pkl.dump(egs_config, open(os.path.join(config.save_dir, 'egs.config'), 'wb'))


if __name__ == "__main__":
    run()

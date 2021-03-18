#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""

@author: samiksadhu
"""

'Computing Joint Histogram of features and Labels'

import argparse
import numpy as np
import bisect
import os
import kaldi_io
import pickle as pkl


def get_minmax(feat_dict):
    feat_min = +np.inf
    feat_max = -np.inf
    for key in feat_dict:
        one_max = np.max(feat_dict[key])
        one_min = np.min(feat_dict[key])
        if one_max > feat_max:
            feat_max = one_max
        if one_min < feat_min:
            feat_min = one_min

    return feat_min, feat_max


def get_signal_label_joint_distribution(alis, feats, minmax_ali, minmax_feat, feat_dim=80, num_bins=100):
    mnx_a = pkl.load(open(minmax_ali, 'rb'))
    mnx_f = pkl.load(open(minmax_feat, 'rb'))
    mn_a, mx_a = mnx_a['min'], mnx_a['max']
    mn_f, mx_f = mnx_f['min'], mnx_f['max']
    sig_bins = np.linspace(mn_f, mx_f, num_bins + 1)
    dist = np.zeros((feat_dim, num_bins, mx_a))
    nums = len(list(feats.keys()))
    count = 0
    for key in feats:
        count += 1
        print('Processing {:f} % of files'.format(count * 100 / nums))
        # print('min_ali={:d} and max_ali={:d}'.format(np.min(alis[key]), np.max(alis[key])))
        # print('min_f={:f} and max_f={:f}'.format(np.min(feats[key]), np.max(feats[key])))
        for idx, label in enumerate(alis[key]):
            f = feats[key][idx, :]
            for r in range(feat_dim):
                ii = int(bisect.bisect_left(sig_bins, f[r]))
                jj = label - 1
                # print('mn_f={:f} and mx_f={:f}'.format(mn_f, mx_f))
                # print('ii={:d} and jj={:d}'.format(ii,jj))
                if ii == 0:
                    ii = 1
                if ii == num_bins + 1:
                    ii = num_bins
                ii = ii - 1
                dist[r, ii, jj] += 1

    return dist


def get_signal_trans_joint_distribution(alis, feats, minmax_ali, minmax_feat, feat_dim=80, num_bins=100):
    mnx_f = pkl.load(open(minmax_feat, 'rb'))
    mn_f, mx_f = mnx_f['min'], mnx_f['max']
    sig_bins = np.linspace(mn_f, mx_f, num_bins + 1)
    dist = np.zeros((feat_dim, num_bins, 2))
    nums = len(list(feats.keys()))
    count = 0
    for key in feats:
        count += 1
        print('Processing {:f} % of files'.format(count * 100 / nums))
        for idx, label in enumerate(alis[key]):
            f = feats[key][idx, :]
            for r in range(feat_dim):
                ii = int(bisect.bisect_left(sig_bins, f[r]))
                jj = int(label)
                if ii == 0:
                    ii = 1
                if ii == num_bins + 1:
                    ii = num_bins
                ii = ii - 1
                dist[r, ii, jj] += 1

    return dist


def get_phoneme_labels(ali_dir):
    ali_files = []
    all_ali_dirs = ali_dir.split(',')
    for ali_dir in all_ali_dirs:
        ali_files.extend([os.path.join(ali_dir, f) for f in os.listdir(ali_dir) if f.startswith('ali.')])

    pdf_ali_dict = {}

    for file in ali_files:
        pdf_ali_file = "ark:ali-to-phones --per-frame {} ark:'gunzip -c {} |' ark:- |".format(
            os.path.join(ali_dir, "final.mdl"),
            file)
        pdf_ali_dict.update({u: d for u, d in kaldi_io.read_vec_int_ark(pdf_ali_file)})

    return pdf_ali_dict


def get_transitions(alis):
    trans_dict = {}
    for utt in alis:
        one_ali = alis[utt]
        r_prev = 2222222
        one_trans = np.zeros(one_ali.shape[0])
        for idx, r in enumerate(one_ali):
            if idx > 0:
                if r_prev != r:
                    # There is a transition here
                    one_trans[idx] = 1
                    one_trans[idx - 1] = 1
                    one_trans[idx + 1] = 1
            r_prev = r
        trans_dict[utt] = one_trans

    return trans_dict


def get_feats(feat_scp):
    return {uttid: feats for uttid, feats in kaldi_io.read_mat_scp(feat_scp)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute Signal-Label Histogram')
    parser.add_argument('scp', help='Feature scp file')
    parser.add_argument('phoneme_ali_dir', help='Phoneme alignment directory')
    parser.add_argument('minmax_ali', help='Alignmnet minmax file')
    parser.add_argument('minmax_feat', help='Feature minmax file')
    parser.add_argument('out_file', help='Output file')
    parser.add_argument("--feat_size", type=int, default=80, help="Feature size")
    parser.add_argument("--analyze_transitions", action="store_true", help="Set to compute MI at transitions")
    args = parser.parse_args()

    all_alis = get_phoneme_labels(args.phoneme_ali_dir)
    if args.analyze_transitions:
        all_alis = get_transitions(all_alis)
    feats = get_feats(args.scp)

    if args.analyze_transitions:
        dist = get_signal_trans_joint_distribution(all_alis, feats, args.minmax_ali, args.minmax_feat,
                                                   feat_dim=args.feat_size, num_bins=100)
    else:
        dist = get_signal_label_joint_distribution(all_alis, feats, args.minmax_ali, args.minmax_feat,
                                                   feat_dim=args.feat_size, num_bins=100)
    pkl.dump(dist, open(args.out_file + '.pkl', 'wb'))

#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:42:56 2018

@author: samiksadhu
"""

'Computing FDLP Modulation Spectral Features'

import argparse
import io
import numpy as np
import os
from scipy.io.wavfile import read
import subprocess
import scipy.fftpack as freqAnalysis
import sys
import time

from features import getFrames, createFbank, computeLpcFast, computeModSpecFromLpc, addReverb, dict2Ark


def getFeats(args, srate=16000, window=np.hanning):
    wavs = args.scp
    outfile = args.outfile
    add_reverb = args.add_reverb
    coeff_0 = args.coeff_0
    coeff_n = args.coeff_n
    order = args.order
    fduration = args.fduration
    frate = args.frate
    nfilters = args.nfilters
    kaldi_cmd = args.kaldi_cmd

    fbank = createFbank(nfilters, int(2 * fduration * srate), srate)

    coeff_num = coeff_n - coeff_0 + 1

    if args.keep_even:
        temp = np.arange(0, coeff_num)
        if coeff_0 % 2 == 0:
            # It starts from odd coefficients
            feat_len = temp[1::2].shape[0]
        else:
            feat_len = temp[0::2].shape[0]

    else:
        feat_len = coeff_num

    if add_reverb:
        if add_reverb == 'small_room':
            sr_r, rir = read('./RIR/RIR_SmallRoom1_near_AnglA.wav')
            rir = rir[:, 1]
            rir = rir / np.power(2, 15)
        elif add_reverb == 'large_room':
            sr_r, rir = read('./RIR/RIR_LargeRoom1_far_AnglA.wav')
            rir = rir[:, 1]
            rir = rir / np.power(2, 15)
        elif add_reverb == 'clean':
            print('%s: No reverberation added!' % sys.argv[0])
        else:
            raise ValueError('Invalid type of reverberation!')

    with open(wavs, 'r') as fid:
        all_feats = {}

        for line in fid:
            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])

            if inwav[-1] == '|':
                proc = subprocess.run(inwav[:-1], shell=True,
                                      stdout=subprocess.PIPE)
                sr, signal = read(io.BytesIO(proc.stdout))
            else:
                sr, signal = read(inwav)
            assert sr == srate, 'Input file has different sampling rate.'

            # I want to work with numbers from 0 to 1 so.... 
            signal = signal / np.power(2, 15)

            if add_reverb:
                if not add_reverb == 'clean':
                    signal = addReverb(signal, rir)

            time_frames = np.array([frame for frame in
                                    getFrames(signal, srate, frate, fduration, window)])

            cos_trans = freqAnalysis.dct(time_frames) / np.sqrt(2 * int(srate * fduration))

            [frame_num, ndct] = np.shape(cos_trans)

            feats = np.zeros((frame_num, nfilters * feat_len))

            print('%s: Computing Features for file: %s, also %d' % (sys.argv[0], uttid, time_frames.shape[0]))
            sys.stdout.flush()
            for i in range(frame_num):

                each_feat = np.zeros([nfilters, feat_len])
                for j in range(nfilters):
                    filt = fbank[j, 0:-1]
                    band_dct = filt * cos_trans[i, :]
                    xlpc, gg = computeLpcFast(band_dct, order)  # Compute LPC coefficients
                    mod_spec = computeModSpecFromLpc(gg, xlpc, coeff_n)
                    temp2 = mod_spec[coeff_0 - 1:coeff_n] 
                    if args.keep_even:
                        if coeff_0 % 2 == 0:
                            each_feat[j, :] = temp2[1::2]
                        else:
                            each_feat[j, :] = temp2[0::2]
                    else:
                        each_feat[j, :] = temp2

                each_feat = np.reshape(each_feat, (1, nfilters * feat_len))

                feats[i, :] = each_feat

            all_feats[uttid] = feats

        dict2Ark(all_feats, outfile, kaldi_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract FDLP Modulation Spectral Features.')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('--nfilters', type=int, default=15, help='number of filters (15)')
    parser.add_argument('--coeff_0', type=int, default=5, help='starting coefficient')
    parser.add_argument('--coeff_n', type=int, default=30, help='ending coefficient')
    parser.add_argument('--keep_even', action='store_true', help='Keep only even coefficients')
    parser.add_argument('--order', type=int, default=50, help='LPC filter order (50)')
    parser.add_argument('--fduration', type=float, default=0.5, help='Window length (0.5 sec)')
    parser.add_argument('--frate', type=int, default=100, help='Frame rate (100 Hz)')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--set_unity_gain', action='store_true', help='Set LPC gain to 1 (True)')
    parser.add_argument('--kaldi_cmd', help='Kaldi command to use to get ark files')
    args = parser.parse_args()

    start_time = time.time()

    print('%s: Extracting features....' % sys.argv[0])
    sys.stdout.flush()
    getFeats(args)

    time_note = 'Execution Time: {t:.3f} seconds'.format(t=time.time() - start_time)
    print(time_note)
    sys.stdout.flush()

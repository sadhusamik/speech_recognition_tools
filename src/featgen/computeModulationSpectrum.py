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
from scipy.fftpack import fft, ifft

from features import getFrames, createFbank, computeLpcFast, computeModSpecFromLpc, addReverb, dict2Ark, \
    createFbankCochlear


def sq_wind(N):
    return np.ones(N)


def getFeats(args, srate=16000, window=np.hanning):
    wavs = args.scp
    scp_type = args.scp_type
    outfile = args.outfile
    add_reverb = args.add_reverb
    coeff_0 = args.coeff_0
    coeff_n = args.coeff_n
    order = args.order
    fduration = args.fduration
    frate = args.frate
    nfilters = args.nfilters
    kaldi_cmd = args.kaldi_cmd

    # Set up mel-filterbank
    fbank_type = args.fbank_type.strip().split(',')
    if args.complex_modulation:
        dur = int(fduration * srate)
    else:
        dur = int(2 * fduration * srate)

    if fbank_type[0] == "mel":
        if len(fbank_type) < 2:
            raise ValueError('Mel filter bank not configured properly....')
        fbank = createFbank(nfilters, dur, srate, warp_fact=float(fbank_type[1]))
    elif fbank_type[0] == "cochlear":
        if len(fbank_type) < 6:
            raise ValueError('Cochlear filter bank not configured properly....')
        if int(fbank_type[3]) == 1:
            print('%s: Alpha is fixed and will not change as a function of the center frequency...' % sys.argv[0])
        fbank = createFbankCochlear(nfilters, dur, srate, om_w=float(fbank_type[1]),
                                    alp=float(fbank_type[2]), fixed=int(fbank_type[3]), bet=float(fbank_type[4]),
                                    warp_fact=float(fbank_type[5]))
    else:
        raise ValueError('Invalid type of filter bank, use mel or cochlear with proper configuration')
    coeff_num = coeff_n - coeff_0 + 1

    if args.keep_even:
        temp = np.arange(0, coeff_num)
        if coeff_0 % 2 == 0:
            # It starts from odd coefficients
            feat_len = temp[1::2].shape[0]
        else:
            feat_len = temp[0::2].shape[0]

    elif args.complex_modulation:
        if args.absolute_value:
            feat_len = coeff_num
        else:
            feat_len = 2 * coeff_num
    else:
        feat_len = coeff_num

    if args.compensate_noise:
        if args.complex_modulation:
            fmax = coeff_num / (fduration)
            faxis = np.linspace(0, fmax, coeff_n)
        else:
            fmax = coeff_num / (2 * fduration)
            faxis = np.linspace(0, fmax, coeff_n)

    if args.no_window:
        print('%s: Using square windows' % sys.argv[0])
        window = sq_wind

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

            if scp_type == 'wav':
                if inwav[-1] == '|':
                    try:
                        proc = subprocess.run(inwav[:-1], shell=True, stdout=subprocess.PIPE)
                        sr, signal = read(io.BytesIO(proc.stdout))
                        skip_rest = False
                    except Exception:
                        skip_rest = True
                else:
                    try:
                        sr, signal = read(inwav)
                        skip_rest = False
                    except Exception:
                        skip_rest = True

                assert sr == srate, 'Input file has different sampling rate.'
            elif scp_type == 'segment':
                try:
                    cmd = 'wav-copy ' + inwav + ' - '
                    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                    sr, signal = read(io.BytesIO(proc.stdout))
                    skip_rest = False
                except Exception:
                    skip_rest = True
            else:
                raise ValueError('Invalid type of scp type, it should be either wav or segment')

            if not skip_rest:
                # I want to work with numbers from 0 to 1 so....
                # signal = signal / np.power(2, 15)

                if add_reverb:
                    if not add_reverb == 'clean':
                        signal = addReverb(signal, rir)

                time_frames = np.array([frame for frame in
                                        getFrames(signal, srate, frate, fduration, window)])

                if args.complex_modulation:
                    cos_trans = freqAnalysis.ifft(time_frames)
                    cos_trans = cos_trans[:, :int(fduration * srate / 2)]
                else:
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
                        if args.complex_modulation:
                            xlpc, gg = computeLpcFast(band_dct, order, keepreal=False)  # Compute LPC coefficients
                            mod_spec = computeModSpecFromLpc(gg, xlpc, coeff_n)
                            if args.compensate_noise:
                                mod_spec = mod_spec * faxis
                            if args.absolute_value:
                                temp2 = np.abs(mod_spec[coeff_0 - 1:coeff_n])
                            else:
                                temp2 = np.append(np.real(mod_spec[coeff_0 - 1:coeff_n]),
                                                  np.imag(mod_spec[coeff_0 - 1:coeff_n]))
                        else:
                            xlpc, gg = computeLpcFast(band_dct, order)
                            mod_spec = np.real(computeModSpecFromLpc(gg, xlpc, coeff_n))
                            if args.compensate_noise:
                                mod_spec = mod_spec * faxis
                            if args.absolute_value:
                                temp2 = np.abs(mod_spec[coeff_0 - 1:coeff_n])
                            else:
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
    parser.add_argument("--scp_type", default='wav', help="scp type can be 'wav' or 'segment'")
    parser.add_argument('--nfilters', type=int, default=15, help='number of filters (15)')
    parser.add_argument('--coeff_0', type=int, default=5, help='starting coefficient')
    parser.add_argument('--coeff_n', type=int, default=30, help='ending coefficient')
    parser.add_argument('--keep_even', action='store_true', help='Keep only even coefficients')
    parser.add_argument('--order', type=int, default=50, help='LPC filter order (50)')
    parser.add_argument('--fduration', type=float, default=0.5, help='Window length (0.5 sec)')
    parser.add_argument('--frate', type=int, default=100, help='Frame rate (100 Hz)')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--fbank_type', type=str, default='mel,1',
                        help='mel,warp_fact OR cochlear,om_w,alpa,fixed,beta,warp_fact')
    parser.add_argument('--set_unity_gain', action='store_true', help='Set LPC gain to 1 (True)')
    parser.add_argument('--no_window', action='store_true', help='Keeps the square window')
    parser.add_argument('--complex_modulation', action='store_true', help='Computes modulation by fft and not dct')
    parser.add_argument('--compensate_noise', action='store_true', help='Compensate 1/f noise in modulation spectrum')
    parser.add_argument('--absolute_value', action='store_true', help='Compute absolute value of modulation spectrum')
    parser.add_argument('--kaldi_cmd', help='Kaldi command to use to get ark files')
    args = parser.parse_args()

    start_time = time.time()

    print('%s: Extracting features....' % sys.argv[0])
    sys.stdout.flush()
    getFeats(args)

    time_note = 'Execution Time: {t:.3f} seconds'.format(t=time.time() - start_time)
    print(time_note)
    sys.stdout.flush()

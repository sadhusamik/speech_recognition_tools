#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:42:56 2018

@author: samiksadhu
"""

'Computing FDLP Spectrogram'

import argparse
import io
import numpy as np
import os
from scipy.io.wavfile import read
import subprocess
import scipy.fftpack as freqAnalysis
import sys
import time
from scipy.fftpack import fft
from random import randrange
from scipy.signal import convolve
import scipy.stats as stats
from features import getFrames, createFbank, createFbankCochlear, computeLpcFast, computeModSpecFromLpc, addReverb, \
    dict2Ark, load_noise, \
    add_noise_to_wav


def getFeats(args, srate=16000, window=np.hamming):
    wavs = args.scp
    scp_type = args.scp_type
    outfile = args.outfile
    coeff_num = args.coeff_num
    coeff_range = args.coeff_range
    order = args.order
    fduration = args.fduration
    frate = args.frate
    nfilters = args.nfilters
    kaldi_cmd = args.kaldi_cmd
    add_noise = args.add_noise
    add_reverb = args.add_reverb

    if args.lifter_config:
        fid = open(args.lifter_config, 'r')
        lifter_config = fid.readline().strip().split(',')
        lifter_config = np.asarray([float(x) for x in lifter_config])

    # Set up mel-filterbank
    fbank_type = args.fbank_type.strip().split(',')
    if fbank_type[0] == "mel":
        if len(fbank_type) < 2:
            raise ValueError('Mel filter bank not configured properly....')
        fbank = createFbank(nfilters, int(2 * fduration * srate), srate, warp_fact=float(fbank_type[1]))
    elif fbank_type[0] == "cochlear":
        if len(fbank_type) < 6:
            raise ValueError('Cochlear filter bank not configured properly....')
        if int(fbank_type[3]) == 1:
            print('%s: Alpha is fixed and will not change as a function of the center frequency...' % sys.argv[0])
        fbank = createFbankCochlear(nfilters, int(2 * fduration * srate), srate, om_w=float(fbank_type[1]),
                                    alp=float(fbank_type[2]), fixed=int(fbank_type[3]), bet=float(fbank_type[4]),
                                    warp_fact=float(fbank_type[5]))
    else:
        raise ValueError('Invalid type of filter bank, use mel or cochlear with proper configuration')

    # Ignore odd modulations
    if args.odd_mod_zero:
        print('%s: Ignoring odd modulations... ' % sys.argv[0])
    if add_noise:
        if add_noise == "clean" or add_noise == "diff":
            print('%s: No noise added!' % sys.argv[0])
        else:
            noise_info = add_noise.strip().split(',')
            noise = load_noise(noise_info[0])

    if add_reverb:
        if add_reverb == 'small_room':
            sr_r, rir = read('./RIR/RIR_SmallRoom1_near_AnglA.wav')
            rir = rir[:, 1]
            rir = rir / np.power(2, 15)
        elif add_reverb == 'medium_room':
            sr_r, rir = read('./RIR/RIR_MediumRoom1_far_AnglA.wav')
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

    # Set up mask
    coeff_range = coeff_range.split(',')
    lowpass = int(coeff_range[0])
    highpass = int(coeff_range[1])
    mask = []
    for i in range(coeff_num):
        if i >= lowpass and i <= highpass:
            mask.append(1)
        else:
            mask.append(0)
    mask = np.asarray(mask)
    args.overlap_fraction = 1 - args.overlap_fraction

    # Setup modulation weights
    args.gamma_weight = args.gamma_weight.strip().split(',')
    if not args.gamma_weight[0] == "None":
        print('%s: Adding gamma filter on modulation frequencies...' % sys.argv[0])
        x = np.linspace(0, order - 1, order)
        scale = float(args.gamma_weight[0])
        shape = float(args.gamma_weight[1])
        pk_required = float(args.gamma_weight[2])
        res = 2 * fduration
        pk_required = pk_required * res
        pk = (shape - 1) * scale
        loc = -pk + pk_required
        mod_wts = stats.gamma.pdf(x, a=shape, loc=loc, scale=scale) * 3 * scale
    with open(wavs, 'r') as fid:

        all_feats = {}
        if args.write_utt2num_frames:
            all_lens = {}

        for line in fid:
            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])

            if scp_type == 'wav':
                if inwav[-1] == '|':
                    try:
                        proc = subprocess.run(inwav[:-1], shell=True, stdout=subprocess.PIPE)
                        sr, signal = read(io.BytesIO(proc.stdout))
                        skip_rest=False
                    except Exception:
                        skip_rest=True
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

            # I want to work with numbers from 0 to 1 so....
            # signal = signal / np.power(2, 15)

            if not skip_rest:
                if add_noise:
                    if not add_noise == "clean":
                        if add_noise == "diff":
                            a = [1, 2, 3, 2, 0, -2, -5, -2, 0, 2, 3, 2, 1]
                            signal = convolve(signal, a, mode='same')
                        else:
                            signal = add_noise_to_wav(signal, noise, float(noise_info[1]))

                if add_reverb:
                    if not add_reverb == 'clean':
                        signal = addReverb(signal, rir)

                tframes = signal.shape[0]  # Number of samples in the signal

                lfr = 1 / (args.overlap_fraction * fduration)
                time_frames = np.array([frame for frame in
                                        getFrames(signal, srate, lfr, fduration, window)])

                cos_trans = freqAnalysis.dct(time_frames) / np.sqrt(2 * int(srate * fduration))

                [frame_num, ndct] = np.shape(cos_trans)

                feats = np.zeros((nfilters, int(np.ceil(tframes * frate / srate))))
                ptr = int(0)

                print('%s: Computing Features for file: %s' % (sys.argv[0], uttid))
                sys.stdout.flush()

                for i in range(0, frame_num):
                    for j in range(nfilters):
                        filt = fbank[j, 0:-1]
                        band_dct = filt * cos_trans[i, :]
                        xlpc, gg = computeLpcFast(band_dct, order)  # Compute LPC coefficients
                        ms = computeModSpecFromLpc(gg, xlpc, coeff_num)
                        ms = ms * mask
                        if args.lifter_config:
                            ms = ms * lifter_config
                        if not args.gamma_weight[0] == "None":
                            ms = ms * mod_wts
                        if args.odd_mod_zero:
                            ms[1::2] = 0
                        ms = fft(ms, 2 * int(fduration * frate))
                        ms = np.abs(np.exp(ms))
                        kk = int(np.round(fduration * frate))
                        kkb2 = int(np.round(fduration * frate / 2))
                        ms = ms[0:kk] * np.hanning(kk) / window(kk)

                        if i == 0:
                            if feats.shape[1] < kkb2:
                                feats[j, :] += ms[kkb2:kkb2 + feats.shape[1]]
                            else:
                                feats[j, ptr:ptr + kkb2] += ms[kkb2:]
                        elif i == frame_num - 1 or i == frame_num - 2:
                            if ms.shape[0] >= feats.shape[1] - ptr:
                                feats[j, ptr:] += ms[:feats.shape[1] - ptr]
                            else:
                                feats[j, ptr:ptr + kk] += ms
                        else:
                            feats[j, ptr:ptr + kk] += ms

                    kk = int(np.round(fduration * frate * args.overlap_fraction))
                    kkb2 = int(np.round(fduration * frate / 2))
                    if i == 0:
                        ptr = int(ptr + kk - kkb2)
                    else:
                        ptr = int(ptr + kk + randrange(2))

                all_feats[uttid] = np.log(np.clip(feats.T, a_max=None, a_min=0.00000000000001))
                if args.write_utt2num_frames:
                    all_lens[uttid] = feats.shape[1]

        dict2Ark(all_feats, outfile, kaldi_cmd)
        if args.write_utt2num_frames:
            with open(outfile + '.len', 'w+') as file:
                for key, lens in all_lens.items():
                    p = "{:s} {:d}".format(key, lens)
                    file.write(p)
                    file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract FDLP Spectrogram.')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('outfile', help='output file')
    parser.add_argument("--scp_type", default='wav', help="scp type can be 'wav' or 'segment'")
    parser.add_argument('--nfilters', type=int, default=20, help='number of filters (15)')
    parser.add_argument('--coeff_num', type=int, default=50, help='Total Number of coefficients to compute')
    parser.add_argument('--coeff_range', type=str, default='1,20', help="Range of Modulation coefficients to keep")
    parser.add_argument('--order', type=int, default=50, help='LPC filter order (50)')
    parser.add_argument('--fduration', type=float, default=0.5, help='Window length (0.5 sec)')
    parser.add_argument('--frate', type=int, default=100, help='Frame rate (100 Hz)')
    parser.add_argument('--overlap_fraction', type=float, default=0.25, help='Fraction of Overlap for OLA')
    parser.add_argument('--kaldi_cmd', default='copy-feats', help='Kaldi command to use to get ark files')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--fbank_type', type=str, default='mel,1',
                        help='mel,warp_fact OR cochlear,om_w,alpa,fixed,beta,warp_fact')
    parser.add_argument('--odd_mod_zero', action='store_true', help='Ignore the odd modulation coefficients')
    parser.add_argument('--gamma_weight', type=str, default='None', help='Configured as scale,shape,pk')
    parser.add_argument('--lifter_config', type=str, default=None, help='Configuration for general liftering')
    parser.add_argument("--write_utt2num_frames", action="store_true", help="Set to write utt2num_frames")
    parser.add_argument('--add_noise',
                        help='Specify "type of noise, snr", types: babble, buccaneer1, buccaneer2, car, destroyerops, f16, factory1, factory2, m109, machinegun, pink, street, volvo, white')
    args = parser.parse_args()

    start_time = time.time()

    print('%s: Extracting features....' % sys.argv[0])
    sys.stdout.flush()
    getFeats(args)

    time_note = 'Execution Time: {t:.3f} seconds'.format(t=time.time() - start_time)
    print(time_note)
    sys.stdout.flush()

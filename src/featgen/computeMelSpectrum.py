"""
Compute Mel Specral features from wav files

Author: Samik Sadhu
"""

import numpy as np
from features import getFrames, createFbank, spliceFeats, addReverb, add_agwn, load_noise, add_noise_to_wav, \
    get_kaldi_ark
from scipy.fftpack import fft, dct
from scipy.io.wavfile import read
import subprocess
import argparse
import sys
import io


def get_args():
    parser = argparse.ArgumentParser('Extract Mel Energy Features')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('--nfilters', type=int, default=23, help='number of filters (30)')
    parser.add_argument('--fduration', type=float, default=0.02, help='Window length (0.02 sec)')
    parser.add_argument('--frate', type=int, default=100, help='Frame rate (100 Hz)')
    parser.add_argument('--nfft', type=int, default=1024, help='Number of points of computing FFT')
    parser.add_argument('--rir', help='Room impulse response file if adding artificial reverberation')
    parser.add_argument('--add_noise',
                        help='Specify "type of noise, snr", types: babble, buccaneer1, buccaneer2, car, destroyerops, f16, factory1, factory2, m109, machinegun, pink, street, volvo, white')

    return parser.parse_args()


def compute_mel_spectrum(args, srate=16000,
                         window=np.hamming):
    wavs = args.scp
    outfile = args.outfile
    add_noise = args.add_noise
    nfft = args.nfft
    fduration = args.fduration
    frate = args.frate
    nfilters = args.nfilters

    fbank = createFbank(nfilters, nfft, srate)

    if add_noise:
        noise_info = add_noise.strip().split(',')
        noise = load_noise(noise_info[0])

    if args.rir:
        sr_r, rir = read(args.rir)
        rir = rir[:, 1]
        rir = rir / np.power(2, 15)
    else:
        print('%s: No reverberation added!' % sys.argv[0])

    with open(wavs, 'r') as fid:

        all_feats = {}
        for line in fid:
            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])

            print('%s: Computing Features for file: %s' % (sys.argv[0], uttid))
            sys.stdout.flush()

            if inwav[-1] == '|':
                proc = subprocess.run(inwav[:-1], shell=True,
                                      stdout=subprocess.PIPE)
                sr, signal = read(io.BytesIO(proc.stdout))
            else:
                sr, signal = read(inwav)
            assert sr == srate, 'Input file has different sampling rate.'

            signal = signal / np.power(2, 15)

            if add_noise:
                signal = add_noise_to_wav(signal, noise, float(noise_info[1]))

            if args.rir:
                signal = addReverb(signal, rir)

            time_frames = np.array([frame for frame in
                                    getFrames(signal, srate, frate, fduration, window)])

            melEnergy_frames = np.log10(
                np.matmul(np.abs(fft(time_frames, int(nfft / 2 + 1), axis=1)), np.transpose(fbank)))

            all_feats[uttid] = melEnergy_frames

        get_kaldi_ark(all_feats, outfile)


if __name__ == '__main__':
    args = get_args()
    compute_mel_spectrum(args)

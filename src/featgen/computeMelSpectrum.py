"""
Compute Mel Specral features from wav files

Author: Samik Sadhu
"""

import numpy as np
from features import getFrames, createFbank, createFbankCochlear, spliceFeats, addReverb, add_agwn, load_noise, \
    add_noise_to_wav, \
    get_kaldi_ark
from scipy.fftpack import fft, dct
from scipy.io.wavfile import read
import subprocess
import argparse
import sys
import io
from scipy.signal import convolve


def get_args():
    parser = argparse.ArgumentParser('Extract Mel Energy Features')
    parser.add_argument('scp', help='scp file')
    parser.add_argument('outfile', help='output file')
    parser.add_argument("--scp_type", default='wav', help="scp type can be 'wav' or 'segment'")
    parser.add_argument("--spectrum_type", default="log", help="log/power For log spectrum or energy spectrum")
    parser.add_argument('--nfilters', type=int, default=23, help='number of filters (30)')
    parser.add_argument('--fduration', type=float, default=0.02, help='Window length (0.02 sec)')
    parser.add_argument('--frate', type=int, default=100, help='Frame rate (100 Hz)')
    parser.add_argument('--nfft', type=int, default=1024, help='Number of points of computing FFT')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--fbank_type', type=str, default='mel,1',
                        help='mel,warp_fact OR cochlear,om_w,alpa,fixed,beta,warp_fact')
    parser.add_argument("--write_utt2num_frames", action="store_true", help="Set to write utt2num_frames")
    parser.add_argument('--add_noise',
                        help='Specify "type of noise, snr", types: babble, buccaneer1, buccaneer2, car, destroyerops, f16, factory1, factory2, m109, machinegun, pink, street, volvo, white')

    return parser.parse_args()


def compute_mel_spectrum(args, srate=16000,
                         window=np.hamming):
    wavs = args.scp
    scp_type = args.scp_type
    outfile = args.outfile
    add_noise = args.add_noise
    nfft = args.nfft
    fduration = args.fduration
    frate = args.frate
    nfilters = args.nfilters
    add_reverb = args.add_reverb

    # Set up mel-filterbank
    fbank_type = args.fbank_type.strip().split(',')
    if fbank_type[0] == "mel":
        if len(fbank_type) < 2:
            raise ValueError('Mel filter bank not configured properly....')
        fbank = createFbank(nfilters, nfft, srate, warp_fact=float(fbank_type[1]))
    elif fbank_type[0] == "cochlear":
        if len(fbank_type) < 6:
            raise ValueError('Cochlear filter bank not configured properly....')
        if int(fbank_type[3]) == 1:
            print('%s: Alpha is fixed and will not change as a function of the center frequency...' % sys.argv[0])
        fbank = createFbankCochlear(nfilters, nfft, srate, om_w=float(fbank_type[1]),
                                    alp=float(fbank_type[2]), fixed=int(fbank_type[3]), bet=float(fbank_type[4]),
                                    warp_fact=float(fbank_type[5]))
    else:
        raise ValueError('Invalid type of filter bank, use mel or cochlear with proper configuration')

    if add_noise:
        if not add_noise == "clean" and not add_noise == "diff":
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

    with open(wavs, 'r') as fid:

        all_feats = {}
        if args.write_utt2num_frames:
            all_lens = {}

        for line in fid:
            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])

            print('%s: Computing Features for file: %s' % (sys.argv[0], uttid))
            sys.stdout.flush()

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
            # signal = signal / np.power(2, 15)

            if not skip_rest:
                if add_noise:
                    if not add_noise == "clean":
                        if add_noise == "diff":
                            # signal = np.diff(signal)
                            a = [1, 2, 3, 2, 0, -2, -5, -2, 0, 2, 3, 2, 1]
                            signal = convolve(signal, a, mode='same')
                        else:
                            signal = add_noise_to_wav(signal, noise, float(noise_info[1]))

                if add_reverb:
                    if not add_reverb == 'clean':
                        signal = addReverb(signal, rir)

                time_frames = np.array([frame for frame in
                                        getFrames(signal, srate, frate, fduration, window)])

                if args.spectrum_type == "log":
                    melEnergy_frames = np.log10(
                        np.matmul(np.abs(fft(time_frames, nfft, axis=1)[:, :int(nfft / 2 + 1)]), np.transpose(fbank)))
                elif args.spectrum_type == "power":
                    melEnergy_frames = np.power(
                        np.matmul(np.abs(fft(time_frames, nfft, axis=1)[:, :int(nfft / 2 + 1)]), np.transpose(fbank)), 2)
                else:
                    print("Spectrum type not supported! ")
                    sys.exit(1)

                all_feats[uttid] = melEnergy_frames
                if args.write_utt2num_frames:
                    all_lens[uttid] = melEnergy_frames.shape[0]
        get_kaldi_ark(all_feats, outfile)

        if args.write_utt2num_frames:
            with open(outfile + '.len', 'w+') as file:
                for key, lens in all_lens.items():
                    p = "{:s} {:d}".format(key, lens)
                    file.write(p)
                    file.write("\n")


if __name__ == '__main__':
    args = get_args()
    compute_mel_spectrum(args)

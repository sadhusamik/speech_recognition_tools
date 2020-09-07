import argparse
import os

import numpy as np
from chainer import Variable
from chainer import cuda
from chainer import serializers
from tqdm import tqdm
import sys

from chime_data import gen_flist_simu, gen_flist_2ch,\
    gen_flist_real, get_audio_data, get_audio_data_1ch, get_audio_data_with_context
from fgnt.beamforming import gev_wrapper_on_masks
from fgnt.signal_processing import audiowrite, stft, istft
from fgnt.utils import Timer
from fgnt.utils import mkdir_p
from nn_models import BLSTMMaskEstimator, SimpleFWMaskEstimator

parser = argparse.ArgumentParser(description='NN GEV beamforming')
parser.add_argument('flist',
                    help='Name of the flist to process (e.g. tr05_simu)')
parser.add_argument('chime_dir',
                    help='Base directory of the CHiME challenge.')
parser.add_argument('sim_dir',
                    help='Base directory of the CHiME challenge simulated data.')
parser.add_argument('output_dir',
                    help='The directory where the enhanced wav files will '
                         'be stored.')
parser.add_argument('model',
                    help='Trained model file')
parser.add_argument('model_type',
                    help='Type of model (BLSTM or FW)')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--single', '-s', default=0, type=int,
                    help='0 for multi-channel and channel number (1-6) for single channel')
parser.add_argument('--track', '-t', default=6, type=int,
                    help='1, 2 or 6 depending on the data used')
args = parser.parse_args()

# Prepare model
if args.model_type == 'BLSTM':
    model = BLSTMMaskEstimator()
elif args.model_type == 'FW':
    model = SimpleFWMaskEstimator()
else:
    raise ValueError('Unknown model type. Possible are "BLSTM" and "FW"')

serializers.load_hdf5(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

stage = args.flist[:2]
scenario = args.flist.split('_')[-1]

if stage == 'tr' and (args.track == 1 or args.track == 2):
    print("No train data for 1ch track and 2ch track");
    sys.exit(0);

# CHiME data handling
if scenario == 'simu':
    if args.track == 6:
        flist = gen_flist_simu(args.chime_dir, args.sim_dir, stage)
    elif args.track == 2:
        flist = gen_flist_2ch(args.chime_dir, stage, scenario)
    elif args.track == 1:
        flist = list()
        for env in ['caf', 'bus', 'str', 'ped']:
            flist_temp = os.listdir(os.path.join(args.chime_dir, 'audio', '16kHz', 'isolated_1ch_track', '{}05_{}_{}'.format(stage, env, scenario)))
            flist_ext = [i for i in flist_temp if i.endswith('.wav')]
            flist_with_dir = [os.path.join(args.chime_dir, 'audio', '16kHz', 'isolated_1ch_track', '{}05_{}_{}'.format(stage, env, scenario), i) for i in flist_ext]
            flist = flist + flist_with_dir
elif scenario == 'real':
    if args.track == 6:
        flist = gen_flist_real(args.chime_dir, stage)
    elif args.track == 2:
        flist = gen_flist_2ch(args.chime_dir, stage, scenario)
    elif args.track == 1:
        flist = list()
        for env in ['caf', 'bus', 'str', 'ped']:
            flist_temp = os.listdir(os.path.join(args.chime_dir, 'audio', '16kHz', 'isolated_1ch_track', '{}05_{}_{}'.format(stage, env, scenario)))
            flist_ext = [i for i in flist_temp if i.endswith('.wav')]
            flist_with_dir = [os.path.join(args.chime_dir, 'audio', '16kHz', 'isolated_1ch_track', '{}05_{}_{}'.format(stage, env, scenario), i) for i in flist_ext]
            flist = flist + flist_with_dir
else:
    raise ValueError('Unknown flist {}'.format(args.flist))

for env in ['caf', 'bus', 'str', 'ped']:
    mkdir_p(os.path.join(args.output_dir, '{}05_{}_{}'.format(
            stage, env, scenario
    )))

t_io = 0
t_net = 0
t_beamform = 0
# Beamform loop
for cur_line in tqdm(flist):
    with Timer() as t:
        if args.track == 6:
            if scenario == 'simu':
                audio_data = get_audio_data(cur_line)
                context_samples = 0
            elif scenario == 'real':
                audio_data, context_samples = get_audio_data_with_context(
                        cur_line[0], cur_line[1], cur_line[2])
        elif args.track == 2:
            audio_data = get_audio_data(cur_line)
            context_samples = 0
        elif args.track == 1:
            audio_data = get_audio_data_1ch(cur_line)
            context_samples = 0
    t_io += t.msecs
    Y = stft(audio_data, time_dim=1).transpose((1, 0, 2))
    Y_var = Variable(np.abs(Y).astype(np.float32), True)
    if args.gpu >= 0:
        Y_var.to_gpu(args.gpu)
    with Timer() as t:
        N_masks, X_masks = model.calc_masks(Y_var)
        N_masks.to_cpu()
        X_masks.to_cpu()
    t_net += t.msecs

    with Timer() as t:
        if args.single >= 1 or args.track == 1:
            Y_hat = X_masks.data * Y
        elif args.single == 0:
            N_mask = np.median(N_masks.data, axis=1)
            X_mask = np.median(X_masks.data, axis=1)
            Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask)
    t_beamform += t.msecs

    if args.track == 1:
        env = cur_line.split('/')[-1].split('_')[2].split('.')[0]
        filename = os.path.join(args.output_dir, '{}05_{}_{}'.format(stage, env.lower(), scenario), os.path.basename(cur_line))
    else:
        if scenario == 'simu' or args.track == 2:
            wsj_name = cur_line.split('/')[-1].split('_')[1]
            spk = cur_line.split('/')[-1].split('_')[0]
            env = cur_line.split('/')[-1].split('_')[-1]
        elif scenario == 'real':
            wsj_name = cur_line[3]
            spk = cur_line[0].split('/')[-1].split('_')[0]
            env = cur_line[0].split('/')[-1].split('_')[-1]

        filename = os.path.join(
                args.output_dir,
                '{}05_{}_{}'.format(stage, env.lower(), scenario),
                '{}_{}_{}.wav'.format(spk, wsj_name, env.upper())
        )
    if args.track == 1:
        with Timer() as t:
            audiowrite(istft(Y_hat[:,0,:])[int(context_samples):], filename, 16000, True, True)
        t_io += t.msecs
    elif args.single == 0:
        with Timer() as t:
            audiowrite(istft(Y_hat)[int(context_samples):], filename, 16000, True, True)
        t_io += t.msecs
    elif args.single >= 1:
        ch = args.single
        with Timer() as t:
            audiowrite(istft(Y_hat[:,ch-1,:])[int(context_samples):], filename, 16000, True, True)
        t_io += t.msecs

print('Finished')
print('Timings: I/O: {:.2f}s | Net: {:.2f}s | Beamformer: {:.2f}s'.format(
        t_io / 1000, t_net / 1000, t_beamform / 1000
))

import argparse
import json
import logging
import os
import pickle

import numpy as np
from chainer import Variable
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from tqdm import tqdm

from chime_data import prepare_training_data
from fgnt.utils import Timer
from fgnt.utils import mkdir_p
from nn_models import BLSTMMaskEstimator
from nn_models import SimpleFWMaskEstimator

parser = argparse.ArgumentParser(description='NN GEV training')
parser.add_argument('data_dir', help='Directory used for the training data '
                                     'and to store the model file.')
parser.add_argument('model_type',
                    help='Type of model (BLSTM or FW)')
parser.add_argument('--chime_dir', default='',
                    help='Base directory of the CHiME challenge. This is '
                         'used to create the training data. If not specified, '
                         'the data_dir must contain some training data.')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--max_epochs', default=25, type=int,
                    help='Max. number of epochs to train')
parser.add_argument('--patience', default=5, type=int,
                    help='Max. number of epochs to wait for better CV loss')
parser.add_argument('--dropout', default=.5, type=float,
                    help='Dropout probability')
args = parser.parse_args()

log = logging.getLogger('nn_gev')
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(args.data_dir, 'nn-gev.log'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

if args.chime_dir != '':
    log.info(
            'Preparing training data and storing it in {}'.format(
                    args.data_dir))
    prepare_training_data(args.chime_dir, args.data_dir)

flists = dict()
for stage in ['tr', 'dt']:
    with open(
            os.path.join(args.data_dir, 'flist_{}.json'.format(stage))) as fid:
        flists[stage] = json.load(fid)
log.debug('Loaded file lists')

# Prepare model
if args.model_type == 'BLSTM':
    model = BLSTMMaskEstimator()
    model_save_dir = os.path.join(args.data_dir, 'BLSTM_model')
    mkdir_p(model_save_dir)
elif args.model_type == 'FW':
    model = SimpleFWMaskEstimator()
    model_save_dir = os.path.join(args.data_dir, 'FW_model')
    mkdir_p(model_save_dir)
else:
    raise ValueError('Unknown model type. Possible are "BLSTM" and "FW"')

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy
log.debug('Prepared model')

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)


def _create_batch(file, volatile=False):
    with open(os.path.join(args.data_dir, file), 'rb') as fid:
        data = pickle.load(fid)
    IBM_X = Variable(data['IBM_X'], volatile=volatile)
    IBM_N = Variable(data['IBM_N'], volatile=volatile)
    Y = Variable(data['Y_abs'], volatile=volatile)
    if args.gpu >= 0:
        for var in [IBM_X, IBM_N, Y]:
            var.to_gpu(args.gpu)
    return IBM_X, IBM_N, Y


# Learning loop
epoch = 0
exhausted = False
best_epoch = 0
best_cv_loss = np.inf
while (epoch < args.max_epochs and not exhausted):
    log.info('Starting epoch {}. Best CV loss was {} at epoch {}'.format(
            epoch, best_cv_loss, best_epoch
    ))

    # training
    perm = np.random.permutation(len(flists['tr']))
    sum_loss_tr = 0
    t_io = 0
    t_fw = 0
    t_bw = 0
    for i in tqdm(perm, desc='Training epoch {}'.format(epoch)):
        with Timer() as t:
            IBM_X, IBM_N, Y = _create_batch(flists['tr'][i])
        t_io += t.msecs

        model.zerograds()
        with Timer() as t:
            loss = model.train_and_cv(Y, IBM_N, IBM_X, args.dropout)
        t_fw += t.msecs

        with Timer() as t:
            loss.backward()
            optimizer.update()
        t_bw += t.msecs

        sum_loss_tr += float(loss.data)

    # cross-validation
    sum_loss_cv = 0
    for i in tqdm(range(len(flists['dt'])),
                  desc='Cross-validation epoch {}'.format(epoch)):
        IBM_X, IBM_N, Y = _create_batch(flists['dt'][i], volatile=True)
        loss = model.train_and_cv(Y, IBM_N, IBM_X, 0.)
        sum_loss_cv += float(loss.data)

    loss_tr = sum_loss_tr / len(flists['tr'])
    loss_cv = sum_loss_cv / len(flists['dt'])

    log.info(
            'Finished epoch {}. '
            'Mean loss during training/cross-validation: {:.3f}/{:.3f}'.format(
                    epoch, loss_tr, loss_cv))
    log.info('Timings: I/O: {:.2f}s | FW: {:.2f}s | BW: {:.2f}s'.format(
            t_io / 1000, t_fw / 1000, t_bw / 1000))

    if loss_cv < best_cv_loss:
        best_epoch = epoch
        best_cv_loss = loss_cv
        model_file = os.path.join(model_save_dir, 'best.nnet')
        log.info('New best loss during cross-validation. Saving model file '
                 'under {}'.format(model_file))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(os.path.join(model_save_dir, 'mlp.tr'), optimizer)

    if epoch - best_epoch == args.patience:
        exhausted = True
        log.info('Patience exhausted. Stopping training')

    epoch += 1

log.info('Finished!')

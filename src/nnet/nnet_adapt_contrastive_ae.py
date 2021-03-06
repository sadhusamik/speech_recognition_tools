import os
import logging
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
from nnet_models import nnetFeedforward
from datasets import nnetDataset
import subprocess
import pickle
import kaldi_io
from ae_model import autoencoderRNN
import torch.nn.functional as F
import sys
import pickle as pkl


def get_args():
    parser = argparse.ArgumentParser(description="Adapt acoustic model based on Contrastive RNN-AE PM Measure")

    parser.add_argument("model", help="Feedforward pytorch nnet model")
    parser.add_argument("pm", help="RNN AE performance monitoring model")
    parser.add_argument("scp", help="scp file to update model")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("dev_egs", help="Development set egs .pt file to check how the classifier is doing")
    parser.add_argument("cmvn", help="cmvn file for posteriors")

    # Other options
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--lr_factor", type=float, default=1, help="Factor to reduce learning rate by every epoch")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--experiment_name", default="exp_run", type=str, help="Name of this experiment")
    parser.add_argument("--store_path", default="exp_am", type=str, help="Where to save the trained models and logs")
    parser.add_argument("--time_shifts", default='3,5,7', help="Time shifts for generating contrastive loss")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Truncate or pad all sequences to this length")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of utterances to use for adaptation")
    parser.add_argument("--optimizer", default="adam", type=str,
                        help="The gradient descent optimizer (e.g., sgd, adam, etc.)")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Initial learning rate")

    return parser.parse_args()


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def samplewise_mse(x, y):
    return torch.mean(((x - y) ** 2), dim=2)


def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]


def adjust_learning_rate(optimizer, lr, f):
    lr = lr * f
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr


def compute_fer(x, l):
    x = softmax(x)
    preds = np.argmax(x, axis=1)
    err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, l)))) * 100 / float(preds.shape[0])
    return err


def stable_mse(x, y):
    return ((x - y) ** 2).mean()


def get_cmvn(cmvn_file):
    shell_cmd = "copy-matrix --binary=false {:s} - ".format(cmvn_file)
    r = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE)
    r = r.stdout.decode('utf-8').split('\n')

    r_m = r[1].strip().split()
    r_v = r[2].strip().split()
    frame_num = float(r_m[-1])
    means = np.asarray([float(x) / frame_num for x in r_m[0:-1]])
    var = np.asarray([float(x) / frame_num for x in r_v[0:-2]])

    return means, var


def update(config):
    # Load model

    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    model = nnetFeedforward(nnet['feature_dim'] * nnet['num_frames'], nnet['num_layers'], nnet['hidden_dim'],
                            nnet['num_classes'])
    model.load_state_dict(nnet['model_state_dict'])

    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id

        model = model.cuda()

    model_dir = os.path.join(config.store_path, config.experiment_name + '.dir')
    os.makedirs(config.store_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(model_dir, config.experiment_name),
        filemode='w')

    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Model Parameters: ')
    logging.info('Number of Layers: %d' % (nnet['num_layers']))
    logging.info('Hidden Dimension: %d' % (nnet['feature_dim']))
    logging.info('Number of Classes: %d' % (nnet['num_classes']))
    logging.info('Data dimension: %d' % (nnet['feature_dim']))
    logging.info('Number of Frames: %d' % (nnet['num_frames']))
    logging.info('Time shifts: %s' % (config.time_shifts))
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info('Initial Learning Rate: %f ' % (config.learning_rate))

    criterion = nn.MSELoss()
    dev_criterion = nn.CrossEntropyLoss()

    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    else:
        raise NotImplementedError("Learning method not supported for the task")
    lr = config.learning_rate

    ts_list = [int(t) for t in config.time_shifts.split(',')]
    max_ts = max(ts_list)

    # Figure out all feature stuff

    shell_cmd = "cat {:s} | shuf > temp".format(config.scp)
    r = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE)

    feats_config = pickle.load(open(config.egs_config, 'rb'))

    if feats_config['feat_type']:
        feat_type = feats_config['feat_type'].split(',')[0]
        trans_path = feats_config['feat_type'].split(',')[1]

    if feat_type == "pca":
        cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, 'temp')
    elif feat_type == "cmvn":
        cmd = "apply-cmvn {} scp:{} ark:- |".format(trans_path, 'temp')
    else:
        cmd = 'temp'

    if feats_config['concat_feats']:
        context = feats_config['concat_feats'].split(',')
        cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])

    # Load performance monitoring model

    pm_model = torch.load(config.pm, map_location=lambda storage, loc: storage)
    ae_model = autoencoderRNN(pm_model['feature_dim'], pm_model['feature_dim'], pm_model['bn_dim'],
                              pm_model['encoder_num_layers'], pm_model['decoder_num_layers'], pm_model['hidden_dim'])
    ae_model.load_state_dict(pm_model['model_state_dict'])

    if config.use_gpu:
        ae_model.cuda()

    for p in ae_model.parameters():  # Do not update performance monitoring block
        p.requires_grad = False

    mean, _ = get_cmvn(config.cmvn)

    ep_loss_dev = []
    ep_fer_dev = []

    load_chunk = torch.load(config.dev_egs)
    dev_data = load_chunk[:, 0:-1]
    dev_labels = load_chunk[:, -1].long()
    dataset = nnetDataset(dev_data, dev_labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    init_fer = True
    if init_fer:
        # Compute initial performance on dev set
        val_losses = []
        val_fer = []

        for batch_x, batch_l in data_loader:
            if config.use_gpu:
                batch_x = Variable(batch_x).cuda()
                batch_l = Variable(batch_l).cuda()
            else:
                batch_x = Variable(batch_x)
                batch_l = Variable(batch_l)

            _, batch_x = model(batch_x)
            val_loss = dev_criterion(batch_x, batch_l)
            val_losses.append(val_loss.item())

            if config.use_gpu:
                val_fer.append(compute_fer(batch_x.cpu().data.numpy(), batch_l.cpu().data.numpy()))
            else:
                val_fer.append(compute_fer(batch_x.data.numpy(), batch_l.data.numpy()))

        ep_loss_dev.append(np.mean(val_losses))
        ep_fer_dev.append(np.mean(val_fer))

        print_log = "Epoch: -1 update Dev loss: {:.3f} :: Dev FER: {:.2f}".format(
            np.mean(val_losses),
            np.mean(val_fer))

        logging.info(print_log)

    for epoch in range(config.epochs):

        if config.use_gpu:
            batch = torch.empty(0, config.max_seq_len, pm_model['feature_dim']).cuda()
        else:
            batch = torch.empty(0, config.max_seq_len, pm_model['feature_dim'])

        lens = []
        utt_count = 0
        update_num = 0

        val_losses = []
        val_fer = []

        train_losses_pos = []
        train_losses_neg = []

        for utt_id, mat in kaldi_io.read_mat_ark(cmd):
            model.eval()
            if config.use_gpu:
                out = model(Variable(torch.FloatTensor(mat)).cuda())
            else:
                out = model(Variable(torch.FloatTensor(mat)))

            if config.use_gpu:
                post = out[1] - torch.FloatTensor(mean).cuda()
            else:
                post = out[1] - torch.FloatTensor(mean)

            lens.append(min(post.shape[0], config.max_seq_len))
            post = F.pad(post, (0, 0, 0, config.max_seq_len - post.size(0)))
            batch = torch.cat([batch, post[None, :, :]], 0)
            utt_count += 1
            sys.stdout.flush()

            if utt_count == config.batch_size:
                update_num += 1
                #### DO THE ADAPTATION

                lens = torch.IntTensor(lens)
                _, indices = torch.sort(lens, descending=True)
                batch_x = batch[indices]
                batch_l = lens[indices]

                outputs = ae_model(batch_x, batch_l)

                # First positive loss
                mse_pos = samplewise_mse(outputs[:, max_ts:-max_ts - 1, :], batch_x[:, max_ts:-max_ts - 1, :])

                # Now find negative loss
                s = outputs[:, max_ts:-max_ts - 1, :].size()

                if config.use_gpu:
                    mse_neg = torch.zeros(s[0], s[1]).cuda()
                else:
                    mse_neg = torch.zeros(s[0], s[1])

                count = 0
                for t in ts_list:
                    count += 1

                    mse_neg += samplewise_mse(outputs[:, max_ts:-max_ts - 1, :],
                                              batch_x[:, max_ts + t:-max_ts - 1 + t, :])
                    mse_neg += samplewise_mse(outputs[:, max_ts:-max_ts - 1, :],
                                              batch_x[:, max_ts - t:-max_ts - 1 - t, :])

                mse_neg = mse_neg / (2*count)

                loss = mse_pos.mean()  # (mse_pos / mse_neg).mean()
                train_losses_pos.append(mse_pos.mean().item())
                train_losses_neg.append(mse_neg.mean().item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if config.use_gpu:
                    batch = torch.empty(0, config.max_seq_len, pm_model['feature_dim']).cuda()
                else:
                    batch = torch.empty(0, config.max_seq_len, pm_model['feature_dim'])
                lens = []
                utt_count = 0

                # CHECK IF ADAPTATION IS WORKING AT ALL

                for batch_x, batch_l in data_loader:
                    if config.use_gpu:
                        batch_x = Variable(batch_x).cuda()
                        batch_l = Variable(batch_l).cuda()
                    else:
                        batch_x = Variable(batch_x)
                        batch_l = Variable(batch_l)

                    _, batch_x = model(batch_x)
                    val_loss = dev_criterion(batch_x, batch_l)
                    val_losses.append(val_loss.item())

                    if config.use_gpu:
                        val_fer.append(compute_fer(batch_x.cpu().data.numpy(), batch_l.cpu().data.numpy()))
                    else:
                        val_fer.append(compute_fer(batch_x.data.numpy(), batch_l.data.numpy()))

        ep_loss_dev.append(np.mean(val_losses))
        ep_fer_dev.append(np.mean(val_fer))

        print_log = "Epoch: {:d} update, Tr +ve MSE Loss: {:.3f} :: Tr -ve MSE Loss: {:.3f} :: Dev loss: {:.3f} :: Dev FER: {:.2f}".format(
            epoch,
            np.mean(train_losses_pos),
            np.mean(train_losses_neg),
            np.mean(val_losses),
            np.mean(val_fer))

        logging.info(print_log)

        torch.save(ep_loss_dev, open(os.path.join(model_dir, "dev_epoch{:d}.loss".format(epoch + 1)), 'wb'))
        torch.save(ep_fer_dev, open(os.path.join(model_dir, "dev_epoch{:d}.fer".format(epoch + 1)), 'wb'))

        # Change learning rate to half
        optimizer, lr = adjust_learning_rate(optimizer, lr, config.lr_factor)
        logging.info('Learning rate changed to {:f}'.format(lr))


if __name__ == '__main__':
    config = get_args()
    update(config)

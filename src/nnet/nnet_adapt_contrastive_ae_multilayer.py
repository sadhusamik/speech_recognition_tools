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


def get_args():
    parser = argparse.ArgumentParser(
        description="Adapt acoustic model based on Contrastive RNN-AE PM Measure on multiple layers")

    parser.add_argument("model", help="Feedforward pytorch nnet model")
    parser.add_argument("pms", help="List of RNN AE performance monitoring model for different layers")
    parser.add_argument("scp", help="scp file to update model")
    parser.add_argument("egs_config", help="config file for generating examples")
    parser.add_argument("dev_egs", help="Development set egs .pt file to check how the classifier is doing")
    parser.add_argument("cmvns", help="All global cmvn files for output from different layers")

    # Other options
    parser.add_argument("--override_trans_path", default=None,
                        help="Override the feature transformation file used in egs config")
    parser.add_argument("--pm_index", default="0,-1,-2,-3,-4",
                        help="Index of pms passed as number of layers from output layer")
    parser.add_argument("--neg_weight", default=1, type=float, help="Weight for the neagtive cost")
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
    parser.add_argument("--loss", default="L1", help="Loss function L1/MSE")

    return parser.parse_args()


def samplewise_mse(x, y):
    return torch.mean(((x - y) ** 2), dim=2)


def samplewise_abs(x, y):
    return torch.mean(torch.abs(x - y), dim=2)


def stable_mse(x, y):
    return ((x - y) ** 2).mean()


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


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
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Using %s loss function' % config.loss)
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info('Negative loss weight: %f' % (config.neg_weight))
    logging.info('Contrastive time shifts: %s' % (config.time_shifts))
    logging.info('Initial Learning Rate: %f ' % (config.learning_rate))

    if config.loss == "MSE":
        criterion = samplewise_mse
    elif config.loss == "L1":
        criterion = samplewise_abs
    else:
        logging.info('Loss function {:s} is not supported'.format(config.loss))
        sys.exit(1)

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
    pi = [int(t) for t in config.pm_index.split(',')]
    ts_list = [int(t) for t in config.time_shifts.split(',')]
    max_ts = max(ts_list)

    # Figure out all feature stuff
    shuff_file = str(os.getpid()) + '.scp'
    shell_cmd = "cat {:s} | shuf > {:s}".format(config.scp, shuff_file)
    r = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE)

    feats_config = pickle.load(open(config.egs_config, 'rb'))

    if feats_config['feat_type']:
        feat_type = feats_config['feat_type'].split(',')[0]
        trans_path = feats_config['feat_type'].split(',')[1]

    if config.override_trans_path is not None:
        trans_path = config.override_trans_path

    if feat_type == "pca":
        cmd = "transform-feats {} scp:{} ark:- |".format(trans_path, shuff_file)
    elif feat_type == "cmvn":
        cmd = "apply-cmvn {} scp:{} ark:- |".format(trans_path, shuff_file)
    else:
        cmd = shuff_file

    if feats_config['concat_feats']:
        context = feats_config['concat_feats'].split(',')
        cmd += " splice-feats --left-context={:s} --right-context={:s} ark:- ark:- |".format(context[0], context[1])

    # Load performance monitoring models
    pm_paths = config.pms.split(',')
    if len(pi) != len(pm_paths):
        logging.error("Number of pm models {:d} and number indices {:d} do not match".format(len(pm_paths),len(pi)))
        sys.exit(0)

    pm_models = []
    feat_dims = []
    for path in pm_paths:

        pm_model = torch.load(path, map_location=lambda storage, loc: storage)
        ae_model = autoencoderRNN(pm_model['feature_dim'], pm_model['feature_dim'], pm_model['bn_dim'],
                                  pm_model['encoder_num_layers'], pm_model['decoder_num_layers'],
                                  pm_model['hidden_dim'])
        ae_model.load_state_dict(pm_model['model_state_dict'])
        feat_dims.append(pm_model['feature_dim'])

        if config.use_gpu:
            ae_model.cuda()

        for p in ae_model.parameters():  # Do not update performance monitoring block
            p.requires_grad = False

        pm_models.append(ae_model)

    cmvn_paths = config.cmvns.split(',')
    means = []
    for path in cmvn_paths:
        mean, _ = get_cmvn(path)
        means.append(mean)

    if len(cmvn_paths) != len(pm_paths):
        logging.error("Number of cmvn paths not equal to number of model paths, exiting training!")
        sys.exit(1)
    else:
        num_pm_models = len(pm_paths)

    ep_loss_dev = []
    ep_fer_dev = []

    load_chunk = torch.load(config.dev_egs)
    dev_data = load_chunk[:, 0:-1]
    dev_labels = load_chunk[:, -1].long()
    dataset = nnetDataset(dev_data, dev_labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=50000, shuffle=True)

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

        batches = []
        for idx in range(num_pm_models):
            if config.use_gpu:
                batch = torch.empty(0, config.max_seq_len, feat_dims[idx]).cuda()
            else:
                batch = torch.empty(0, config.max_seq_len, feat_dims[idx])
            batches.append(batch)

        lens = []
        utt_count = 0
        update_num = 0

        val_losses = []
        val_fer = []
        tr_losses = []
        for idx in range(num_pm_models):
            tr_losses.append([])

        for utt_id, mat in kaldi_io.read_mat_ark(cmd):

            lens.append(min(mat.shape[0], config.max_seq_len))

            # Get model output
            optimizer.zero_grad()
            if config.use_gpu:
                out = model(Variable(torch.FloatTensor(mat)).cuda())
            else:
                out = model(Variable(torch.FloatTensor(mat)))

            # Get batches for update

            for idx in range(num_pm_models):
                if config.use_gpu:
                    if pi[idx] == 0:
                        post = out[1] - Variable(torch.FloatTensor(means[0])).cuda()
                    else:
                        post = out[0][pi[idx]] - Variable(torch.FloatTensor(means[idx])).cuda()
                else:
                    if pi[idx] == 0:
                        post = out[1] - Variable(torch.FloatTensor(means[0]))
                    else:
                        post = out[0][pi[idx]] - Variable(torch.FloatTensor(means[idx]))

                post = F.pad(post, (0, 0, 0, config.max_seq_len - post.size(0)))
                batch = batches[idx]
                batch = torch.cat([batch, post[None, :, :]], 0)
                batches[idx] = batch

            utt_count += 1

            if utt_count == config.batch_size:
                update_num += 1

                ## DO THE ADAPTATION

                lens = torch.IntTensor(lens)
                _, indices = torch.sort(lens, descending=True)

                if config.use_gpu:
                    loss_all = torch.FloatTensor([1]).cuda()
                else:
                    loss_all = torch.FloatTensor([1])
                for idx in range(num_pm_models):
                    batch_x = batches[idx][indices]
                    ae_model = pm_models[idx]
                    batch_l = lens[indices]

                    outputs = ae_model(batch_x, batch_l)

                    # First positive loss
                    mse_pos = criterion(outputs[:, max_ts:-max_ts - 1, :], batch_x[:, max_ts:-max_ts - 1, :])

                    # Now find negative loss
                    s = outputs[:, max_ts:-max_ts - 1, :].size()

                    if config.use_gpu:
                        mse_neg = torch.zeros(s[0], s[1]).cuda()
                    else:
                        mse_neg = torch.zeros(s[0], s[1])

                    count = 0
                    for t in ts_list:
                        count += 1

                        mse_neg += criterion(outputs[:, max_ts:-max_ts - 1, :],
                                             batch_x[:, max_ts + t:-max_ts - 1 + t, :])
                        mse_neg += criterion(outputs[:, max_ts:-max_ts - 1, :],
                                             batch_x[:, max_ts - t:-max_ts - 1 - t, :])

                    mse_neg = (mse_neg * config.neg_weight) / (2 * count)
                    loss = (mse_pos / mse_neg).mean()
                    loss_all *= loss

                    tl = tr_losses[idx]
                    tl.append(loss.item())
                    tr_losses[idx] = tl

                loss_all.backward()
                optimizer.step()

                batches = []
                for idx in range(num_pm_models):
                    if config.use_gpu:
                        batch = torch.empty(0, config.max_seq_len, feat_dims[idx]).cuda()
                    else:
                        batch = torch.empty(0, config.max_seq_len, feat_dims[idx])
                    batches.append(batch)
                lens = []
                utt_count = 0

        logging.info("Finished unsupervised adaptation for epoch {:d} with multi-layer RNN-AE Loss".format(epoch))

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

        print_log = "Epoch: {:d} update ".format(epoch)
        for idx in range(num_pm_models):
            print_log = print_log + "Tr loss layer {:d} = {:.3f} | ".format(pi[idx], np.mean(tr_losses[idx]))

        print_log = print_log + "Dev loss: {:.3f} | Dev FER: {:.2f}".format(np.mean(val_losses), np.mean(val_fer))

        logging.info(print_log)

        model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch + 1) + '.model')
        torch.save({
            'epoch': epoch + 1,
            'ep_loss_dev': ep_loss_dev,
            'ep_fer_dev': ep_fer_dev,
            'tr_losses': tr_losses,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))

        # Change learning rate to half
        optimizer, lr = adjust_learning_rate(optimizer, lr, config.lr_factor)
        logging.info('Learning rate changed to {:f}'.format(lr))


if __name__ == '__main__':
    config = get_args()
    update(config)

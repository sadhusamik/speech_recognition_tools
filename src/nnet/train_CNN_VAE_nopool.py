import os
import logging
import argparse
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
from nnet_models_cnn import nnetVAECNNNopool
from datasets import nnetDatasetSeq

import subprocess


def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def compute_fer(x, l):
    x = softmax(x)
    preds = np.argmax(x, axis=1)
    err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, l)))) * 100 / float(preds.shape[0])
    return err


def vae_loss(x, ae_out, latent_out):
    log_lhood = torch.mean(-0.5 * torch.pow((x - ae_out), 2) - 0.5 * np.log(2 * np.pi * 1))
    kl_loss = 0.5 * torch.mean(
        1 - torch.pow(latent_out[0], 2) - torch.pow(torch.exp(latent_out[1]), 2) + 2 * latent_out[1])
    return log_lhood, kl_loss


def vae_loss_nopool(x, ae_out, latent_out, batch_l):
    x = torch.transpose(x, 1, 2)
    ae_out = torch.transpose(ae_out, 1, 2)

    log_lhood = torch.mean(
        -0.5 * torch.pow((pad2list(x, batch_l) - pad2list(ae_out, batch_l)), 2) - 0.5 * np.log(2 * np.pi * 1))

    latent_out = (pad2list(torch.transpose(latent_out[0], 1, 2), batch_l),
                  pad2list(torch.transpose(latent_out[1], 1, 2), batch_l))

    kl_loss = 0.5 * torch.mean(
        1 - torch.pow(latent_out[0], 2) - torch.pow(torch.exp(latent_out[1]), 2) + 2 * latent_out[1])

    return log_lhood, kl_loss


def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN VAE without pooling")

    parser.add_argument("egs_dir", type=str, help="Path to the preprocessed data")
    parser.add_argument("store_path", type=str, help="Where to save the trained models and logs")

    parser.add_argument("--in_channels", default="1,32,64", help="Input channels")
    parser.add_argument("--out_channels", default="32,64,128", help="Output channels")
    parser.add_argument("--kernel", default="3,5", help="Kernel size (height,width) NOTE: They must be odd numbers")


    # Training configuration
    parser.add_argument("--optimizer", default="adam", type=str,
                        help="The gradient descent optimizer (e.g., sgd, adam, etc.)")
    parser.add_argument("--batch_size", default=64, type=int, help="Training minibatch size")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--train_set", default="train_si284", help="Name of the training datatset")
    parser.add_argument("--dev_set", default="test_dev93", help="Name of development dataset")
    parser.add_argument("--clip_thresh", type=float, default=1, help="Gradient clipping threshold")
    parser.add_argument("--lrr", type=float, default=0.5, help="Learning rate reduction rate")
    parser.add_argument("--lr_tol", type=float, default=0.01,
                        help="Percentage of tolerance to leave on dev error for lr scheduling")
    parser.add_argument("--weight_decay", type=float, default=0, help="L2 Regularization weight")

    # Misc configurations
    parser.add_argument("--feature_dim", default=13, type=int, help="The dimension of the input and predicted frame")
    parser.add_argument("--num_frames", default=512, type=int, help="Number of frames in each utterance")
    parser.add_argument("--model_save_interval", type=int, default=10,
                        help="Number of epochs to skip before every model save")
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--load_data_workers", default=10, type=int, help="Number of parallel data loaders")
    parser.add_argument("--experiment_name", default="exp_run", type=str, help="Name of this experiment")

    return parser.parse_args()


def run(config):
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
    logging.info('Input Channels: %s' % (config.in_channels))
    logging.info('Output Channels: %s' % (config.out_channels))
    logging.info('Kernel: %s' % (config.kernel))
    logging.info('Data dimension: %d' % (config.feature_dim))
    logging.info('Bottleneck dimension: %d' % (config.bn_dim))
    logging.info('Number of Frames: %d' % (config.num_frames))
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info('Initial Learning Rate: %f ' % (config.learning_rate))
    logging.info('Learning rate reduction rate: %f ' % (config.lrr))
    logging.info('Weight decay: %f ' % (config.weight_decay))

    sys.stdout.flush()

    in_channels = [int(x) for x in config.in_channels.split(',')]
    out_channels = [int(x) for x in config.out_channels.split(',')]
    kernel = tuple([int(x) for x in config.kernel.split(',')])

    model = nnetVAECNNNopool(config.feature_dim, config.num_frames, in_channels,
                             out_channels, kernel, config.bn_dim, config.use_gpu)

    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id

        model = model.cuda()

    lr = config.learning_rate

    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError("Learning method not supported for the task")

    model_path = os.path.join(model_dir, config.experiment_name + '__epoch_0.model')
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))

    ep_vae_rec_tr = []
    ep_vae_kl_tr = []
    ep_vae_rec_dev = []
    ep_vae_kl_dev = []

    # Load Datasets

    dataset_train = nnetDatasetSeq(os.path.join(config.egs_dir, config.train_set))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)

    dataset_dev = nnetDatasetSeq(os.path.join(config.egs_dir, config.dev_set))
    data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=config.batch_size, shuffle=True)

    err_p = 0
    best_model_state = None

    for epoch_i in range(config.epochs):

        ####################
        ##### Training #####
        ####################

        model.train()
        train_vae_rec_losses = []
        train_vae_kl_losses = []

        # Main training loop

        for batch_x, batch_l, lab in data_loader_train:
            batch_x = batch_x[:, None, :, :]  # change the data format for CNNs
            batch_x = torch.transpose(batch_x, 2, 3)
            if config.use_gpu:
                batch_x = Variable(batch_x).cuda()
                batch_l = Variable(batch_l).cuda()
            else:
                batch_x = Variable(batch_x)
                batch_l = Variable(batch_l)

            optimizer.zero_grad()

            # Main forward pass
            ae_out, latent_out = model(batch_x)

            loss = vae_loss_nopool(batch_x[:, 0, :, :], ae_out[:, 0, :, :], latent_out, batch_l)

            train_vae_rec_losses.append(loss[0].item())
            train_vae_kl_losses.append(loss[1].item())

            (-loss[0] - loss[1]).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_thresh)
            optimizer.step()

        ep_vae_rec_tr.append(np.mean(train_vae_rec_losses))
        ep_vae_kl_tr.append(np.mean(train_vae_kl_losses))

        ######################
        ##### Validation #####
        ######################

        model.eval()

        with torch.set_grad_enabled(False):

            val_vae_rec_losses = []
            val_vae_kl_losses = []

            for batch_x, batch_l, lab in data_loader_dev:
                batch_x = batch_x[:, None, :, :]  # change the data format for CNNs
                batch_x = torch.transpose(batch_x, 2, 3)

                if config.use_gpu:
                    batch_x = Variable(batch_x).cuda()
                    batch_l = Variable(batch_l).cuda()
                else:
                    batch_x = Variable(batch_x)
                    batch_l = Variable(batch_l)

                # Main forward pass
                ae_out, latent_out = model(batch_x)

                loss = vae_loss_nopool(batch_x[:, 0, :, :], ae_out[:, 0, :, :], latent_out, batch_l)

                val_vae_rec_losses.append(loss[0].item())
                val_vae_kl_losses.append(loss[1].item())

            ep_vae_rec_dev.append(np.mean(val_vae_rec_losses))
            ep_vae_kl_dev.append(np.mean(val_vae_kl_losses))

        # Manage learning rate
        if epoch_i == 0:
            err_p = -np.mean(val_vae_rec_losses) - np.mean(val_vae_kl_losses)
            best_model_state = model.state_dict()
        else:
            if -np.mean(val_vae_rec_losses) - np.mean(val_vae_kl_losses) > (100 - config.lr_tol) * err_p / 100:
                logging.info(
                    "Val loss went up, Changing learning rate from {:.6f} to {:.6f}".format(lr, config.lrr * lr))
                lr = config.lrr * lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                model.load_state_dict(best_model_state)
            else:
                err_p = -np.mean(val_vae_rec_losses) - np.mean(val_vae_kl_losses)
                best_model_state = model.state_dict()

        print_log = "Epoch: {:d} ((lr={:.6f})) Tr VAE ll={:.3f},rec={:.3f},kld={:.3f} :: Val VAE ll={:.3f},rec={:.3f},kld={:.3f}".format(
            epoch_i + 1, lr,
            ep_vae_kl_tr[-1] + ep_vae_rec_tr[-1], ep_vae_rec_tr[-1], ep_vae_kl_tr[-1],
            ep_vae_kl_dev[-1] + ep_vae_rec_dev[-1], ep_vae_rec_dev[-1], ep_vae_kl_dev[-1])

        logging.info(print_log)

        if (epoch_i + 1) % config.model_save_interval == 0:
            model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
        torch.save({
            'epoch': epoch_i + 1,
            'feature_dim': config.feature_dim,
            'num_frames': config.num_frames,
            'in_channels': config.in_channels,
            'out_channels': config.out_channels,
            'kernel': config.kernel,
            'bn_dim': config.bn_dim,
            'ep_vae_kl_tr': ep_vae_kl_tr,
            'ep_vae_rec_tr': ep_vae_rec_tr,
            'ep_vae_kl_dev': ep_vae_kl_dev,
            'ep_vae_rec_dev': ep_vae_rec_dev,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))


if __name__ == '__main__':
    config = get_args()
    run(config)

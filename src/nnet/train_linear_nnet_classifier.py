import os
import logging
import argparse
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
from nnet_models import nnetLinearWithConv
from datasets import nnetDatasetSeq
import pickle as pkl

import subprocess


def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]


def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def compute_fer(x, l):
    x = softmax(x)
    preds = np.argmax(x, axis=1)
    err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, l)))) * 100 / float(preds.shape[0])
    return err


def get_args():
    parser = argparse.ArgumentParser(description="Train Linear Feedforward Acoustic Model")

    parser.add_argument("egs_dir", type=str, help="Path to the preprocessed data")
    parser.add_argument("store_path", type=str, help="Where to save the trained models and logs")

    parser.add_argument("--num_layers", default=5, type=int, help="Number of layers")
    parser.add_argument("--hidden_dim", default=512, type=int, help="Number of hidden nodes")

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
    parser.add_argument("--lr_tol", type=float, default=0.5,
                        help="Percentage of tolerance to leave on dev error for lr scheduling")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="L2 Regularization weight")

    # Misc configurations
    parser.add_argument("--num_classes", type=int, default=42,
                        help="Number of phonetic/state classes for acoustic model")
    parser.add_argument("--model_save_interval", type=int, default=50,
                        help="Number of epochs to skip before every model save")
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--feature_dim", default=13, type=int, help="The dimension of the input and predicted frame")
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

    # Load feature configuration
    egs_config = pkl.load(open(os.path.join(config.egs_dir, config.train_set, 'egs.config'), 'rb'))
    context = egs_config['concat_feats'].split(',')
    num_frames = int(context[0]) + int(context[1]) + 1

    logging.info('Model Parameters: ')
    logging.info('Number of Layers: %d' % (config.num_layers))
    logging.info('Hidden Dimension: %d' % (config.feature_dim))
    logging.info('Number of Classes: %d' % (config.num_classes))
    logging.info('Data dimension: %d' % (config.feature_dim))
    logging.info('Number of Frames: %d' % (num_frames))
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info('Initial Learning Rate: %f ' % (config.learning_rate))
    logging.info('Dropout: %f ' % (config.dropout))
    logging.info('Learning rate reduction rate: %f ' % (config.lrr))
    logging.info('Weight decay: %f ' % (config.weight_decay))
    sys.stdout.flush()

    model = nnetLinearWithConv(config.feature_dim * num_frames, config.num_layers, config.hidden_dim, config.num_classes)

    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id

        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

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

    # Load datasets
    dataset_train = nnetDatasetSeq(os.path.join(config.egs_dir, config.train_set))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)

    dataset_dev = nnetDatasetSeq(os.path.join(config.egs_dir, config.dev_set))
    data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=config.batch_size, shuffle=True)

    model_path = os.path.join(model_dir, config.experiment_name + '__epoch_0.model')
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))

    ep_loss_tr = []
    ep_fer_tr = []
    ep_loss_dev = []
    ep_fer_dev = []
    err_p = 0
    best_model_state = None
    for epoch_i in range(config.epochs):

        ####################
        ##### Training #####
        ####################

        model.train()
        train_losses = []
        tr_fer = []
        # Main training loop
        for batch_x, batch_l, lab in data_loader_train:
            _, indices = torch.sort(batch_l, descending=True)
            if config.use_gpu:
                batch_x = Variable(batch_x[indices]).cuda()
                batch_l = Variable(batch_l[indices]).cuda()
                lab = Variable(lab[indices]).cuda()
            else:
                batch_x = Variable(batch_x[indices])
                batch_l = Variable(batch_l[indices])
                lab = Variable(lab[indices])

            optimizer.zero_grad()
            # Main forward pass
            class_out = model(batch_x, batch_l)
            class_out = pad2list(class_out, batch_l)
            lab = pad2list(lab, batch_l)

            loss = criterion(class_out, lab)

            train_losses.append(loss.item())
            if config.use_gpu:
                tr_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
            else:
                tr_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_thresh)
            optimizer.step()

        ep_loss_tr.append(np.mean(train_losses))
        ep_fer_tr.append(np.mean(tr_fer))

        ######################
        ##### Validation #####
        ######################

        model.eval()
        val_losses = []
        val_fer = []
        # Main training loop
        for batch_x, batch_l, lab in data_loader_dev:
            _, indices = torch.sort(batch_l, descending=True)
            if config.use_gpu:
                batch_x = Variable(batch_x[indices]).cuda()
                batch_l = Variable(batch_l[indices]).cuda()
                lab = Variable(lab[indices]).cuda()
            else:
                batch_x = Variable(batch_x[indices])
                batch_l = Variable(batch_l[indices])
                lab = Variable(lab[indices])

            optimizer.zero_grad()
            # Main forward pass
            class_out = model(batch_x, batch_l)
            class_out = pad2list(class_out, batch_l)
            lab = pad2list(lab, batch_l)

            loss = criterion(class_out, lab)

            val_losses.append(loss.item())
            if config.use_gpu:
                val_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
            else:
                val_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

        # Manage learning rate and revert model
        if epoch_i == 0:
            err_p = np.mean(val_losses)
            best_model_state = model.state_dict()
        else:
            if np.mean(val_losses) > (100 - config.lr_tol) * err_p / 100:
                logging.info(
                    "Val loss went up, Changing learning rate from {:.6f} to {:.6f}".format(lr, config.lrr * lr))
                lr = config.lrr * lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                model.load_state_dict(best_model_state)
            else:
                err_p = np.mean(val_losses)
                best_model_state = model.state_dict()

        ep_loss_dev.append(np.mean(val_losses))
        ep_fer_dev.append(np.mean(val_fer))

        print_log = "Epoch: {:d} ((lr={:.6f})) Tr loss: {:.3f} :: Tr FER: {:.2f}".format(epoch_i + 1, lr,
                                                                                         ep_loss_tr[-1],
                                                                                         ep_fer_tr[-1])
        print_log += " || Val: {:.3f} :: Val FER: {:.2f}".format(ep_loss_dev[-1], ep_fer_dev[-1])
        logging.info(print_log)

        if (epoch_i + 1) % config.model_save_interval == 0:
            model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
            torch.save({
                'epoch': epoch_i + 1,
                'feature_dim': config.feature_dim,
                'num_frames': num_frames,
                'num_classes': config.num_classes,
                'num_layers': config.num_layers,
                'hidden_dim': config.hidden_dim,
                'ep_loss_tr': ep_loss_tr,
                'ep_loss_dev': ep_loss_dev,
                'dropout': config.dropout,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))


if __name__ == '__main__':
    config = get_args()
    run(config)

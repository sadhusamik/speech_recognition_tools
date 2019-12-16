import os
import logging
import argparse
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
from nnet_models import nnetFeedforward
from datasets import nnetDataset

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


def get_args():
    parser = argparse.ArgumentParser(description="Train Feedforward Acoustic Model")

    parser.add_argument("--num_layers", default=5, type=int, help="Number of layers")
    parser.add_argument("--hidden_dim", default=512, type=int, help="Number of hidden nodes")

    # Training configuration
    parser.add_argument("--optimizer", default="adam", type=str,
                        help="The gradient descent optimizer (e.g., sgd, adam, etc.)")
    parser.add_argument("--batch_size", default=64, type=int, help="Training minibatch size")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--train_set", default="train_si284", help="Name of the training datatset")
    parser.add_argument("--dev_sets", default="test_dev93", help="Name of development dataset")

    # Misc configurations
    parser.add_argument("--num_frames", type=int, default=1, help="Number of context frames")
    parser.add_argument("--num_classes", type=int, default=42,
                        help="Number of phonetic/state classes for acoustic model")
    parser.add_argument("--model_save_interval", type=int, default=50,
                        help="Number of epochs to skip before every model save")
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--feature_dim", default=13, type=int, help="The dimension of the input and predicted frame")
    parser.add_argument("--load_data_workers", default=10, type=int, help="Number of parallel data loaders")
    parser.add_argument("--experiment_name", default="exp_run", type=str, help="Name of this experiment")
    parser.add_argument("--store_path", default="exp_am", type=str, help="Where to save the trained models and logs")
    parser.add_argument("--egs_dir", default="exp/am_model", type=str, help="Path to the preprocessed data")

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
    logging.info('Number of Layers: %d' % (config.num_layers))
    logging.info('Hidden Dimension: %d' % (config.feature_dim))
    logging.info('Number of Classes: %d' % (config.num_classes))
    logging.info('Data dimension: %d' % (config.feature_dim))
    logging.info('Number of Frames: %d' % (config.num_frames))
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info('Initial Learning Rate: %f ' % (config.learning_rate))
    sys.stdout.flush()

    model = nnetFeedforward(config.feature_dim * config.num_frames, config.num_layers, config.hidden_dim,
                            config.num_classes)

    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id

        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

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

    # setup tensorboard logger
    # tensorboard_logger.configure(
    #    os.path.join(model_dir, config.experiment_name + '.tb_log'))

    train_chunks = [os.path.join(config.egs_dir, config.train_set, x) for x in
                    os.listdir(os.path.join(config.egs_dir, config.train_set)) if x.endswith('.pt')]

    dev_sets = config.dev_sets.split(",")

    for x in dev_sets:
        logging.info('Using Dev set: %s' % (x))

    sys.stdout.flush()

    all_val_chunks = {}
    for d in dev_sets:
        all_val_chunks[d] = [os.path.join(config.egs_dir, d, x) for x in os.listdir(os.path.join(config.egs_dir, d)) if
                             x.endswith('.pt')]

    model_path = os.path.join(model_dir, config.experiment_name + '__epoch_0.model')
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))

    ep_loss_tr = []
    ep_fer_tr = []
    ep_loss_dev = {}
    ep_fer_dev = {}
    for x in dev_sets:
        ep_loss_dev[x] = []
        ep_fer_dev[x] = []

    for epoch_i in range(config.epochs):

        ####################
        ##### Training #####
        ####################

        model.train()
        train_losses = []
        tr_fer = []
        # Main training loop
        for chunk in train_chunks:
            load_chunk = torch.load(chunk)
            train_data = load_chunk[:, 0:-1]
            train_labels = load_chunk[:, -1].long()
            dataset = nnetDataset(train_data, train_labels)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

            for batch_x, batch_l in data_loader:
                if config.use_gpu:
                    batch_x = Variable(batch_x).cuda()
                    batch_l = Variable(batch_l).cuda()
                else:
                    batch_x = Variable(batch_x)
                    batch_l = Variable(batch_l)

                batch_x = model(batch_x)
                optimizer.zero_grad()
                loss = criterion(batch_x, batch_l)

                train_losses.append(loss.item())
                if config.use_gpu:
                    tr_fer.append(compute_fer(batch_x.cpu().data.numpy(), batch_l.cpu().data.numpy()))
                else:
                    tr_fer.append(compute_fer(batch_x.data.numpy(), batch_l.data.numpy()))

                loss.backward()
                optimizer.step()

        ep_loss_tr.append(np.mean(train_losses))
        ep_fer_tr.append(np.mean(tr_fer))

        ######################
        ##### Validation #####
        ######################

        model.eval()

        with torch.set_grad_enabled(False):

            for x in dev_sets:
                val_losses = []
                val_fer = []
                val_chunks = all_val_chunks[x]

                for chunk in val_chunks:

                    load_chunk = torch.load(chunk)
                    dev_data = load_chunk[:, 0:-1]
                    dev_labels = load_chunk[:, -1].long()
                    dataset = nnetDataset(dev_data, dev_labels)
                    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

                    for batch_x, batch_l in data_loader:
                        if config.use_gpu:
                            batch_x = Variable(batch_x).cuda()
                            batch_l = Variable(batch_l).cuda()
                        else:
                            batch_x = Variable(batch_x)
                            batch_l = Variable(batch_l)

                        batch_x = model(batch_x)
                        optimizer.zero_grad()
                        val_loss = criterion(batch_x, batch_l)
                        val_losses.append(val_loss.item())
                        if config.use_gpu:
                            val_fer.append(compute_fer(batch_x.cpu().data.numpy(), batch_l.cpu().data.numpy()))
                        else:
                            val_fer.append(compute_fer(batch_x.data.numpy(), batch_l.data.numpy()))

                ep_loss_d = ep_loss_dev[x]
                ep_loss_d.append(np.mean(val_losses))
                ep_loss_dev[x] = ep_loss_d

                ep_fer_d = ep_fer_dev[x]
                ep_fer_d.append(np.mean(val_fer))
                ep_fer_dev[x] = ep_fer_d

        print_log = "Epoch: {:d} Tr loss: {:.3f} :: Tr FER: {:.2f}".format(epoch_i + 1, ep_loss_tr[-1], ep_fer_tr[-1])

        for x in dev_sets:
            print_log += " || Val ({}): {:.3f} :: Tr FER: {:.2f}".format(x, ep_loss_dev[x][-1], ep_fer_dev[x][-1])

        logging.info(print_log)

        torch.save(ep_loss_tr, open(os.path.join(model_dir, "tr_epoch{:d}.loss".format(epoch_i + 1)), 'wb'))
        torch.save(ep_loss_dev, open(os.path.join(model_dir, "dev_epoch{:d}.loss".format(epoch_i + 1)), 'wb'))

        if (epoch_i + 1) % config.model_save_interval == 0:
            model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
            torch.save({
                'epoch': epoch_i + 1,
                'feature_dim': config.feature_dim,
                'num_frames': config.num_frames,
                'num_classes': config.num_classes,
                'num_layers': config.num_layers,
                'hidden_dim': config.hidden_dim,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))


if __name__ == '__main__':
    config = get_args()
    run(config)

import os
import logging
import argparse
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
from nnet_models import nnetAEClassifierMultitask
from datasets import nnetDatasetSeq
import pickle as pkl

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


def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Feedforward Acoustic Model and an Autoencoder with multitask training")

    parser.add_argument("egs_dir", type=str, help="Path to the preprocessed data")
    parser.add_argument("store_path", type=str, help="Where to save the trained models and logs")

    parser.add_argument("--encoder_num_layers", default=3, type=int, help="Number of encoder layers")
    parser.add_argument("--classifier_num_layers", default=1, type=int, help="Number of decoder layers")
    parser.add_argument("--ae_num_layers", default=1, type=int, help="Number of decoder layers")
    parser.add_argument("--hidden_dim", default=512, type=int, help="Number of hidden nodes")
    parser.add_argument("--bn_dim", default=30, type=int, help="Bottle neck dim")

    # Training configuration
    parser.add_argument("--optimizer", default="adam", type=str,
                        help="The gradient descent optimizer (e.g., sgd, adam, etc.)")
    parser.add_argument("--batch_size", default=64, type=int, help="Training minibatch size")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--train_set", default="train_si284", help="Name of the training datatset")
    parser.add_argument("--dev_set", default="test_dev93", help="Name of development dataset")
    parser.add_argument("--ae_loss", default="MSE", help="Loss function L1/MSE")

    # Misc configurations
    parser.add_argument("--num_classes", type=int, default=42,
                        help="Number of phonetic/state classes for acoustic model")
    parser.add_argument("--feature_dim", default=13, type=int, help="The dimension of the input and predicted frame")
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

    # Load feature configuration
    egs_config = pkl.load(open(os.path.join(config.egs_dir, 'egs.config'), 'rb'))
    context = egs_config['concat_feats'].split(',')
    num_frames = int(context[0]) + int(context[1]) + 1

    logging.info('Model Parameters: ')
    logging.info('Encoder Number of Layers: %d' % (config.encoder_num_layers))
    logging.info('Classifier Number of Layers: %d' % (config.classifier_num_layers))
    logging.info('AE Number of Layers: %d' % (config.ae_num_layers))
    logging.info('Hidden Dimension: %d' % (config.feature_dim))
    logging.info('Number of Classes: %d' % (config.num_classes))
    logging.info('Data dimension: %d' % (config.feature_dim))
    logging.info('Bottleneck dimension: %d' % (config.bn_dim))
    logging.info('Number of Frames: %d' % (num_frames))
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info('Initial Learning Rate: %f ' % (config.learning_rate))
    sys.stdout.flush()

    model = nnetAEClassifierMultitask(config.feature_dim * num_frames, config.num_classes, config.encoder_num_layers,
                                      config.classifier_num_layers, config.ae_num_layers,
                                      config.hidden_dim, config.bn_dim)
    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id

        model = model.cuda()

    criterion_classifier = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

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

    model_path = os.path.join(model_dir, config.experiment_name + '__epoch_0.model')
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))

    ep_loss_tr = []
    ep_fer_tr = []
    ep_ae_tr = []
    ep_loss_dev = []
    ep_fer_dev = []
    ep_ae_dev = []

    # Load Datasets

    dataset_train = nnetDatasetSeq(os.path.join(config.egs_dir, config.train_set))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)

    dataset_dev = nnetDatasetSeq(os.path.join(config.egs_dir, config.dev_set))
    data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=config.batch_size, shuffle=True)

    for epoch_i in range(config.epochs):

        ####################
        ##### Training #####
        ####################

        model.train()
        train_losses = []
        train_ae_losses = []
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

            # Main forward pass
            class_out, ae_out = model(batch_x, batch_l)

            # Convert all the weird tensors to frame-wise form
            class_out = pad2list(class_out, batch_l)
            batch_x = pad2list(batch_x, batch_l)
            ae_out = pad2list(ae_out, batch_l)
            lab = pad2list(lab, batch_l)

            optimizer.zero_grad()

            loss_classifier = criterion_classifier(class_out, lab)
            loss_ae = criterion_ae(ae_out, batch_x)
            loss = loss_classifier + loss_ae

            train_losses.append(loss_classifier.item())
            train_ae_losses.append(loss_ae.item())
            if config.use_gpu:
                tr_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
            else:
                tr_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

            loss.backward()
            optimizer.step()

        ep_loss_tr.append(np.mean(train_losses))
        ep_fer_tr.append(np.mean(tr_fer))
        ep_ae_tr.append(np.mean(train_ae_losses))

        ######################
        ##### Validation #####
        ######################

        model.eval()

        with torch.set_grad_enabled(False):

            val_losses = []
            val_ae_losses = []
            val_fer = []

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

                # Main forward pass
                class_out, ae_out = model(batch_x, batch_l)

                # Convert all the weird tensors to frame-wise form
                class_out = pad2list(class_out, batch_l)
                batch_x = pad2list(batch_x, batch_l)
                ae_out = pad2list(ae_out, batch_l)
                lab = pad2list(lab, batch_l)

                loss_classifier = criterion_classifier(class_out, lab)
                loss_ae = criterion_ae(ae_out, batch_x)

                val_losses.append(loss_classifier.item())
                val_ae_losses.append(loss_ae.item())

                if config.use_gpu:
                    val_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
                else:
                    val_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

            ep_loss_dev.append(np.mean(val_losses))
            ep_fer_dev.append(np.mean(val_fer))
            ep_ae_dev.append(np.mean(val_ae_losses))

        print_log = "Epoch: {:d} Tr loss: {:.3f} :: Tr FER: {:.2f}".format(epoch_i + 1, ep_loss_tr[-1], ep_fer_tr[-1])
        print_log += " || Val : {:.3f} :: Val FER: {:.2f}".format(ep_loss_dev[-1], ep_fer_dev[-1])
        print_log += " || AE Loss (Train) : {:.3f} :: AE Loss (Dev) : {:.3f} ".format(ep_ae_tr[-1], ep_ae_dev[-1])
        logging.info(print_log)

        if (epoch_i + 1) % config.model_save_interval == 0:
            model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
            torch.save({
                'epoch': epoch_i + 1,
                'feature_dim': config.feature_dim,
                'num_frames': num_frames,
                'num_classes': config.num_classes,
                'encoder_num_layers': config.encoder_num_layers,
                'classifier_num_layers': config.classifier_num_layers,
                'ae_num_layers': config.ae_num_layers,
                'hidden_dim': config.hidden_dim,
                'bn_dim': config.bn_dim,
                'ep_loss_tr': ep_loss_tr,
                'ep_loss_dev': ep_loss_dev,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))


if __name__ == '__main__':
    config = get_args()
    run(config)

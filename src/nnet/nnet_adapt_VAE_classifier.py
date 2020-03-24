import os
import logging
import argparse
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
from nnet_models import nnetVAEClassifier
from datasets import nnetDatasetSeqAE, nnetDatasetSeq
from os import listdir
import pickle
import random

import subprocess


def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]


def vae_loss(x, ae_out, latent_out):
    log_lhood = torch.mean(-0.5 * torch.pow((x - ae_out), 2) - 0.5 * np.log(2 * np.pi * 1))
    kl_loss = 0.5 * torch.mean(
        1 - torch.pow(latent_out[0], 2) - torch.pow(torch.exp(latent_out[1]), 2) + 2 * latent_out[1])
    return log_lhood, kl_loss


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def compute_fer(x, l):
    x = softmax(x)
    preds = np.argmax(x, axis=1)
    err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, l)))) * 100 / float(preds.shape[0])
    return err


def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


def sym_kld(X, Y):
    return torch.sum(X * (torch.log(X) - torch.log(Y))) / X.size(0) + torch.sum(
        Y * (torch.log(Y) - torch.log(X))) / X.size(0)


def mmeasure_loss(X, del_list=[5, 25, 45, 65], use_gpu=True):
    sm = nn.Softmax(dim=1)
    kld = nn.KLDivLoss()
    X = sm(X)
    if use_gpu:
        m_acc = torch.FloatTensor([0]).cuda()
    else:
        m_acc = torch.FloatTensor([0])

    for d in del_list:
        m_acc += sym_kld(X[d:, :], X[0:-d, :]) + kld(X[0:-d:, :], X[d:, :])
    return m_acc / len(del_list)


def get_args():
    parser = argparse.ArgumentParser(
        description="Adapt VAE classifier model")

    parser.add_argument("model", help="Feedforward pytorch nnet model")
    parser.add_argument("egs_dir", type=str, help="Path to the preprocessed data")
    parser.add_argument("store_path", type=str, help="Where to save the trained models and logs")

    # Training configuration
    parser.add_argument("--optimizer", default="adam", type=str,
                        help="The gradient descent optimizer (e.g., sgd, adam, etc.)")
    parser.add_argument("--batch_size", default=64, type=int, help="Training minibatch size")
    parser.add_argument("--learning_rate", default=0.0005, type=float, help="Initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--anchor_set", default="train_anchor", help="Anchor Set for Adaptation")
    parser.add_argument("--adapt_set", default="train_adapt", help="Adaptation Set")
    parser.add_argument("--test_set", default="test", help="Test Set")
    parser.add_argument("--clip_thresh", type=float, default=1, help="Gradient clipping threshold")
    parser.add_argument("--mm_weight", type=float, default=1, help="Weight for mmeasure loss component")
    parser.add_argument("--adapt_weight", type=float, default=100, help="Weight for adapt set AE loss component")
    parser.add_argument("--anchor_weight", type=float, default=1,
                        help="Weight for anchor set AE and cross entropy loss component")

    # Misc configurations
    parser.add_argument("--model_save_interval", type=int, default=50,
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

    # Load model
    nnet = torch.load(config.model, map_location=lambda storage, loc: storage)
    model = nnetVAEClassifier(nnet['feature_dim'] * nnet['num_frames'], nnet['num_classes'],
                              nnet['encoder_num_layers'], nnet['classifier_num_layers'], nnet['ae_num_layers'],
                              nnet['hidden_dim'],
                              nnet['bn_dim'], 0.5, config.use_gpu)
    model.load_state_dict(nnet['model_state_dict'])

    # I want to only update the encoder
    for p in model.classifier.parameters():
        p.requires_grad = False

    for p in model.vae_decoder.parameters():
        p.requires_grad = False

    logging.info('Model Parameters: ')
    logging.info('Encoder Number of Layers: %d' % (nnet['encoder_num_layers']))
    logging.info('Classifier Number of Layers: %d' % (nnet['classifier_num_layers']))
    logging.info('AE Number of Layers: %d' % (nnet['ae_num_layers']))
    logging.info('Hidden Dimension: %d' % (nnet['hidden_dim']))
    logging.info('Number of Classes: %d' % (nnet['num_classes']))
    logging.info('Data dimension: %d' % (nnet['feature_dim']))
    logging.info('Bottleneck dimension: %d' % (nnet['bn_dim']))
    logging.info('Number of Frames: %d' % (nnet['num_frames']))
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info('Initial Learning Rate: %f ' % (config.learning_rate))
    logging.info('Encoder Dropout: %f ' % (nnet['enc_dropout']))
    sys.stdout.flush()

    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id

        model = model.cuda()

    criterion_classifier = nn.CrossEntropyLoss()

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

    ep_vae_adapt = []
    ep_mm_adapt = []
    ep_loss_anchor = []
    ep_fer_anchor = []
    ep_vae_anchor = []
    ep_loss_test = []
    ep_fer_test = []
    ep_vae_test = []

    # Load Datasets

    # Anchor set
    path = os.path.join(config.egs_dir, config.anchor_set)
    with open(os.path.join(path, 'lengths.pkl'), 'rb') as f:
        lengths_anchor = pickle.load(f)
    labels_anchor = torch.load(os.path.join(path, 'labels.pkl'))
    anchor_ids = list(labels_anchor.keys())

    # Adaptation Set
    dataset_adapt = nnetDatasetSeqAE(os.path.join(config.egs_dir, config.adapt_set))
    data_loader_adapt = torch.utils.data.DataLoader(dataset_adapt, batch_size=config.batch_size, shuffle=True)

    # Test Set
    dataset_test = nnetDatasetSeq(os.path.join(config.egs_dir, config.test_set))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=True)

    # Start off with initial performance on test set

    model.eval()
    test_losses = []
    test_vae_losses = []
    test_fer = []
    for batch_x, batch_l, lab in data_loader_test:

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
        class_out, ae_out, latent_out = model(batch_x, batch_l)

        # Convert all the weird tensors to frame-wise form
        class_out = pad2list(class_out, batch_l)
        batch_x = pad2list(batch_x, batch_l)
        lab = pad2list(lab, batch_l)

        ae_out = pad2list(ae_out, batch_l)
        latent_out = (pad2list(latent_out[0], batch_l), pad2list(latent_out[1], batch_l))

        loss_classifier = criterion_classifier(class_out, lab)
        loss_vae = vae_loss(batch_x, ae_out, latent_out)

        test_losses.append(loss_classifier.item())
        test_vae_losses.append(loss_vae[0].item() + loss_vae[1].item())

        if config.use_gpu:
            test_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
        else:
            test_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

    print_log = "Initial Testset Error : Adapt (Test) loss: {:.3f} :: Adapt (Test) FER: {:.2f} :: Adapt (Test) Vae log-likelihood loss: {:.3f}".format(
        np.mean(test_losses), np.mean(test_fer), np.mean(test_vae_losses))

    logging.info(print_log)

    for epoch_i in range(config.epochs):

        ######################
        ##### Adaptation #####
        ######################

        model.train()
        adapt_vae_losses = []
        adapt_mm_losses = []
        anchor_losses = []
        anchor_vae_losses = []
        anchor_fer = []
        test_losses = []
        test_vae_losses = []
        test_fer = []

        # Main training loop

        for batch_x, batch_l in data_loader_adapt:

            # First do the adaptation

            _, indices = torch.sort(batch_l, descending=True)
            if config.use_gpu:
                batch_x = Variable(batch_x[indices]).cuda()
                batch_l = Variable(batch_l[indices]).cuda()
            else:
                batch_x = Variable(batch_x[indices])
                batch_l = Variable(batch_l[indices])

            # Main forward pass
            optimizer.zero_grad()
            class_out, ae_out, latent_out = model(batch_x, batch_l)

            # Convert all the weird tensors to frame-wise form
            class_out = pad2list(class_out, batch_l)
            batch_x = pad2list(batch_x, batch_l)

            ae_out = pad2list(ae_out, batch_l)
            latent_out = (pad2list(latent_out[0], batch_l), pad2list(latent_out[1], batch_l))

            loss_vae = vae_loss(batch_x, ae_out, latent_out)
            mm_loss = mmeasure_loss(class_out, use_gpu=config.use_gpu)
            loss = config.adapt_weight * (
                    -loss_vae[0] - loss_vae[1]) - config.mm_weight * mm_loss  # Just the autoencoder loss
            adapt_vae_losses.append(loss_vae[0].item() + loss_vae[1].item())
            adapt_mm_losses.append(mm_loss.item())

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_thresh)
            optimizer.step()

            # Now lets try to anchor the parameters as close as possible to previously seen data

            # Select anchor data randomly
            ids = [random.choice(anchor_ids) for i in range(config.batch_size)]
            batch_x = torch.cat([torch.load(os.path.join(path, index))[None, :, :] for index in ids])
            batch_l = torch.cat([torch.IntTensor([lengths_anchor[index]]) for index in ids])
            lab = torch.cat([labels_anchor[index][None, :] for index in ids])

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
            optimizer.zero_grad()
            class_out, ae_out, latent_out = model(batch_x, batch_l)

            # Convert all the weird tensors to frame-wise form
            class_out = pad2list(class_out, batch_l)
            batch_x = pad2list(batch_x, batch_l)
            lab = pad2list(lab, batch_l)

            ae_out = pad2list(ae_out, batch_l)
            latent_out = (pad2list(latent_out[0], batch_l), pad2list(latent_out[1], batch_l))

            loss_classifier = criterion_classifier(class_out, lab)
            loss_vae = vae_loss(batch_x, ae_out, latent_out)
            loss = config.anchor_weight * (
                    -loss_vae[0] - loss_vae[1] + loss_classifier)  # Use all the loss for anchor set

            anchor_losses.append(loss_classifier.item())
            anchor_vae_losses.append(loss_vae[0].item() + loss_vae[1].item())

            if config.use_gpu:
                anchor_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
            else:
                anchor_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_thresh)
            optimizer.step()

        ## Test it on the WSJ test set

        model.eval()

        for batch_x, batch_l, lab in data_loader_test:

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
            class_out, ae_out, latent_out = model(batch_x, batch_l)

            # Convert all the weird tensors to frame-wise form
            class_out = pad2list(class_out, batch_l)
            batch_x = pad2list(batch_x, batch_l)
            lab = pad2list(lab, batch_l)

            ae_out = pad2list(ae_out, batch_l)
            latent_out = (pad2list(latent_out[0], batch_l), pad2list(latent_out[1], batch_l))

            loss_classifier = criterion_classifier(class_out, lab)
            loss_vae = vae_loss(batch_x, ae_out, latent_out)

            test_losses.append(loss_classifier.item())
            test_vae_losses.append(loss_vae[0].item() + loss_vae[1].item())

            if config.use_gpu:
                test_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
            else:
                test_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

        ep_vae_adapt.append(np.mean(adapt_vae_losses))
        ep_mm_adapt.append(np.mean(adapt_mm_losses))

        ep_loss_anchor.append(np.mean(anchor_losses))
        ep_fer_anchor.append(np.mean(anchor_fer))
        ep_vae_anchor.append(np.mean(anchor_vae_losses))

        ep_loss_test.append(np.mean(test_losses))
        ep_fer_test.append(np.mean(test_fer))
        ep_vae_test.append(np.mean(test_vae_losses))
        print_log = "Epoch: {:d} Adapt (Test) loss: {:.3f} :: Adapt (Test) FER: {:.2f}".format(epoch_i + 1,
                                                                                               ep_loss_test[-1],
                                                                                               ep_fer_test[-1])

        print_log += " || Anchor loss : {:.3f} :: Anchor FER: {:.2f}".format(ep_loss_anchor[-1], ep_fer_anchor[-1])

        print_log += " || VAE llhood (Adapt) : {:.3f} :: VAE llhood (Anchor) : {:.3f} :: VAE llhood (Test) : {:.3f} ".format(
            ep_vae_adapt[-1],
            ep_vae_anchor[-1], ep_vae_test[-1])

        print_log += " || Adapt mm loss : {:.3f} ".format(ep_mm_adapt[-1])

        logging.info(print_log)

        if (epoch_i + 1) % config.model_save_interval == 0:
            model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
            torch.save({
                'epoch': epoch_i + 1,
                'feature_dim': nnet['feature_dim'],
                'num_frames': nnet['num_frames'],
                'num_classes': nnet['num_classes'],
                'encoder_num_layers': nnet['encoder_num_layers'],
                'classifier_num_layers': nnet['classifier_num_layers'],
                'ae_num_layers': nnet['ae_num_layers'],
                'ep_vae_adapt': ep_vae_adapt,
                'ep_mm_adapt': ep_mm_adapt,
                'ep_loss_anchor': ep_loss_anchor,
                'ep_fer_anchor': ep_fer_anchor,
                'ep_vae_anchor': ep_vae_anchor,
                'ep_loss_test': ep_loss_test,
                'ep_fer_test': ep_fer_test,
                'ep_vae_test': ep_vae_test,
                'hidden_dim': nnet['hidden_dim'],
                'bn_dim': nnet['bn_dim'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))


if __name__ == '__main__':
    config = get_args()
    run(config)

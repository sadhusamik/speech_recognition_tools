import os
import logging
import argparse
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
from nnet_models import nnetCurlMultistreamClassifier
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


def curl_loss_supervised(x, ae_out, latent_out, mean_p, idx, use_gpu=True):
    if use_gpu:
        loss = torch.FloatTensor([0]).cuda()
        mean_p = Variable(mean_p).cuda()
    else:
        loss = torch.FloatTensor([0])
        mean_p = Variable(mean_p)

    log_lhood = torch.sum(-0.5 * torch.pow((x - ae_out[idx]), 2) - 0.5 * np.log(2 * np.pi * 1), dim=1)
    kl_loss = 0.5 * torch.sum(
        1 - torch.pow((latent_out[1][idx] - mean_p[idx, :]), 2) - torch.pow(torch.exp(latent_out[2][idx]), 2) + 2 *
        latent_out[2][idx], dim=1)
    loss += torch.mean((log_lhood + kl_loss))

    cat_reg = torch.mean(torch.log(latent_out[0][:, idx]))
    loss += cat_reg
    return loss


def curl_loss_unsupervised(x, ae_out, latent_out, mean_p, use_gpu=True):
    if use_gpu:
        loss = torch.FloatTensor([0]).cuda()
        mean_p = Variable(mean_p).cuda()
    else:
        loss = torch.FloatTensor([0])
        mean_p = Variable(mean_p)

    for idx, out in enumerate(ae_out):
        log_lhood = torch.sum(-0.5 * torch.pow((x - out), 2) - 0.5 * np.log(2 * np.pi * 1), dim=1)
        kl_loss = 0.5 * torch.sum(
            1 - torch.pow((latent_out[1][idx] - mean_p[idx, :]), 2) - torch.pow(torch.exp(latent_out[2][idx]), 2) + 2 *
            latent_out[2][idx], dim=1)
        loss += torch.mean(latent_out[0][:, idx] * (log_lhood + kl_loss))

    cat_reg = torch.mean(torch.sum(latent_out[0] * torch.log(latent_out[0]), dim=1) + np.log(latent_out[0].shape[1]))
    loss -= cat_reg
    return loss


def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


def pad2list3d(padded_seq, lengths):
    return torch.cat([padded_seq[:, i, 0:lengths[i]] for i in range(padded_seq.size(1))], dim=1)


def get_args():
    parser = argparse.ArgumentParser(description="Train a CURL Model and Classifier in multitask fashion (Version 2)")

    parser.add_argument("egs_dir", type=str, help="Path to the preprocessed data")
    parser.add_argument("store_path", type=str, help="Where to save the trained models and logs")

    parser.add_argument("--encoder_num_layers", default=3, type=int, help="Number of encoder layers")
    parser.add_argument("--decoder_num_layers", default=1, type=int, help="Number of decoder layers")
    parser.add_argument("--classifier_num_layers", default=1, type=int, help="Number of classifier layers")
    parser.add_argument("--hidden_dim", default=512, type=int, help="Number of hidden nodes for CURL")
    parser.add_argument("--hidden_dim_classifier", default=512, type=int, help="Number of hidden nodes for classifier")
    parser.add_argument("--bn_dim", default=60, type=int, help="Bottle neck dim")
    parser.add_argument("--comp_num", default=20, type=int, help="Number of GMM components")
    parser.add_argument("--num_classes", default=20, type=int, help="Number of output classes")
    parser.add_argument("--comp_label", default=0, type=int, help="Component index to use for training")

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
    parser.add_argument("--lr_tol", type=float, default=-0.001,
                        help="Percentage of tolerance to leave on dev error for lr scheduling")
    parser.add_argument("--encoder_grad_scale", type=float, default=1, help="Encoder gradient scaling factor")
    parser.add_argument("--weight_decay", type=float, default=0, help="L2 Regularization weight")
    parser.add_argument("--load_checkpoint", type=str, default="None", help="Model checkpoint to load")
    parser.add_argument("--load_previous_model", type=str, default="None", help="Load Model trained on another data")

    # Misc configurations
    parser.add_argument("--feature_dim", default=13, type=int, help="The dimension of the input and predicted frame")
    parser.add_argument("--model_save_interval", type=int, default=10,
                        help="Number of epochs to skip before every model save")
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--load_data_workers", default=10, type=int, help="Number of parallel data loaders")
    parser.add_argument("--experiment_name", default="exp_run", type=str, help="Name of this experiment")

    return parser.parse_args()


def run(config):
    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id

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
    logging.info('Encoder Number of Layers: %d' % (config.encoder_num_layers))
    logging.info('Decoder Number of Layers: %d' % (config.decoder_num_layers))
    logging.info('Classifier Number of Layers: %d' % (config.classifier_num_layers))
    logging.info('Hidden Dimension: %d' % (config.hidden_dim))
    logging.info('Classifier Hidden Dimension: %d' % (config.hidden_dim_classifier))
    logging.info('Data dimension: %d' % (config.feature_dim))
    logging.info('Number of classes: %d' % (config.num_classes))
    logging.info('Bottleneck dimension: %d' % (config.bn_dim))
    logging.info('Component Number: %d' % (config.comp_num))
    logging.info('Number of Frames: %d' % (num_frames))
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info('Initial Learning Rate: %f ' % (config.learning_rate))
    logging.info('Learning rate reduction rate: %f ' % (config.lrr))
    logging.info('Weight decay: %f ' % (config.weight_decay))
    logging.info('Encoder Gradient Scale: %f ' % (config.encoder_grad_scale))

    sys.stdout.flush()

    model = nnetCurlMultistreamClassifier(config.feature_dim * num_frames, config.encoder_num_layers,
                                          config.decoder_num_layers, config.classifier_num_layers, config.hidden_dim,
                                          config.hidden_dim_classifier, config.bn_dim, config.comp_num,
                                          config.num_classes,
                                          config.use_gpu, enc_scale=config.encoder_grad_scale)

    lr = config.learning_rate
    criterion = nn.CrossEntropyLoss()

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

    if config.use_gpu:
        model = model.cuda()

    if config.load_previous_model != "None":
        ckpt = torch.load(config.load_previous_model)
        model.load_state_dict(ckpt["model_state_dict"])
        model.expand_component(config.use_gpu)
        previous_mean_p = torch.from_numpy(ckpt["prior_means"])
        previous_mean_p = torch.cat([previous_mean_p, 5 * (torch.rand(1, config.bn_dim) - 0.5)])

    if config.load_checkpoint != "None":
        ckpt = torch.load(config.load_checkpoint)
        model.load_state_dict(ckpt["model_state_dict"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        ep_start = ckpt["epoch"]
        means_p = torch.from_numpy(ckpt["prior_means"])
    else:
        ep_start = 0
        if config.load_previous_model != "None":
            means_p = previous_mean_p
        else:
            means_p = 5 * (torch.rand(config.comp_num, config.bn_dim) - 0.5)
        model_path = os.path.join(model_dir, config.experiment_name + '__epoch_0.model')
        torch.save({
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'prior_means': means_p.numpy()}, (open(model_path, 'wb')))

    if config.use_gpu:
        model = model.cuda()

    ep_curl_tr = []
    ep_loss_tr = []
    ep_fer_tr = []
    ep_curl_dev = []
    ep_loss_dev = []
    ep_fer_dev = []

    # Load Datasets

    dataset_train = nnetDatasetSeq(os.path.join(config.egs_dir, config.train_set))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)

    dataset_dev = nnetDatasetSeq(os.path.join(config.egs_dir, config.dev_set))
    data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=config.batch_size, shuffle=True)

    err_p = 10000000
    best_model_state = model.state_dict()

    for epoch_i in range(ep_start, config.epochs):

        ####################
        ##### Training #####
        ####################

        model.train()
        train_curl_losses = []
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
            class_out, ae_out, latent_out = model(batch_x, batch_l)

            """
            # Keep only on-zero label rows for label and class_out
            nonzero_idx = []
            for idx, l in enumerate(lab):
                if torch.sum(l) != 0:
                    nonzero_idx.append(idx)
            nonzero_idx = torch.FloatTensor(nonzero_idx).long()
            lab = lab[nonzero_idx]
            class_out = class_out[nonzero_idx]
            """

            # Convert all the weird tensors to frame-wise form
            batch_x = pad2list(batch_x, batch_l)
            ae_out = pad2list3d(ae_out, batch_l)
            # if nonzero_idx.nelement() != 0:
            class_out = pad2list(class_out[config.comp_label], batch_l)
            lab = pad2list(lab, batch_l)

            latent_out = (
                pad2list(latent_out[0], batch_l), pad2list3d(latent_out[1], batch_l),
                pad2list3d(latent_out[2], batch_l))

            # if nonzero_idx.nelement() != 0:
            loss_class = criterion(class_out, lab)
            train_losses.append(loss_class.item())

            loss = curl_loss_supervised(batch_x, ae_out, latent_out, means_p, config.comp_label, use_gpu=config.use_gpu)

            train_curl_losses.append(loss.item())

            # if nonzero_idx.nelement() != 0:
            if config.use_gpu:
                tr_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
            else:
                tr_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))
            # if nonzero_idx.nelement() != 0:
            # (loss_class).backward()
            (-loss + 100 * loss_class).backward()
            # else:
            # (-loss).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_thresh)
            optimizer.step()

        ep_curl_tr.append(np.mean(train_curl_losses))
        ep_loss_tr.append(np.mean(train_losses))
        ep_fer_tr.append(np.mean(tr_fer))

        ######################
        ##### Validation #####
        ######################

        model.eval()

        # with torch.set_grad_enabled(False):
        val_curl_losses = []
        val_losses = []
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

            batch_l = batch_l
            # Main forward pass
            class_out, ae_out, latent_out = model(batch_x, batch_l)

            # Convert all the weird tensors to frame-wise form
            batch_x = pad2list(batch_x, batch_l)
            class_out = pad2list(class_out[config.comp_label], batch_l)
            lab = pad2list(lab, batch_l)
            ae_out = pad2list3d(ae_out, batch_l)
            latent_out = (pad2list(latent_out[0], batch_l), pad2list3d(latent_out[1], batch_l),
                          pad2list3d(latent_out[2], batch_l))

            loss_class = criterion(class_out, lab)
            loss = curl_loss_supervised(batch_x, ae_out, latent_out, means_p, config.comp_label,
                                        use_gpu=config.use_gpu)

            val_curl_losses.append(loss.item())
            val_losses.append(loss_class.item())
            if config.use_gpu:
                val_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
            else:
                val_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

        ep_curl_dev.append(np.mean(val_curl_losses))
        ep_loss_dev.append(np.mean(val_losses))
        ep_fer_dev.append(np.mean(val_fer))

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

        print_log = "Epoch: {:d} ((lr={:.6f})) Tr CURL Log-likelihood: {:.3f} || Tr Loss: {:.3f} || Tr FER: {:.3f} :: Val CURL Log-likelihood: {:.3f} || Val Loss: {:.3f} || Val FER: {:.3f}".format(
            epoch_i + 1, lr, ep_curl_tr[-1], ep_loss_tr[-1], ep_fer_tr[-1], ep_curl_dev[-1], ep_loss_dev[-1],
            ep_fer_dev[-1])

        logging.info(print_log)
        sys.stdout.flush()

        if (epoch_i + 1) % config.model_save_interval == 0:
            model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
            torch.save({
                'epoch': epoch_i + 1,
                'feature_dim': config.feature_dim,
                'num_frames': num_frames,
                'encoder_num_layers': config.encoder_num_layers,
                'decoder_num_layers': config.decoder_num_layers,
                'classifier_num_layers': config.classifier_num_layers,
                'hidden_dim': config.hidden_dim,
                'hidden_dim_classifier': config.hidden_dim_classifier,
                'comp_num': config.comp_num,
                'num_classes': config.num_classes,
                'bn_dim': config.bn_dim,
                'ep_curl_tr': ep_curl_tr,
                'ep_curl_dev': ep_curl_dev,
                'prior_means': means_p.numpy(),
                'lr': lr,
                'encoder_grad_scale': config.encoder_grad_scale,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))


if __name__ == '__main__':
    config = get_args()
    run(config)

# -*- coding: utf-8 -*-
"""
"""
import argparse
import os
import sys
import yaml
import time
import random
from abc import ABCMeta, abstractmethod
from datetime import datetime
try:
    from bpdb import set_trace
except ImportError:
    from pdb import set_trace

from tqdm import tqdm
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from comet_ml import Experiment

from errors import FileNotFoundError, GPUNotFoundError, UnknownOptimizationMethodError, NotSupportedError
#from dataset_indexing.pytorch import RandomNoise

    
class TrainLogger(object):
    """ Logger of training network.

    Args:
        out (str): Output directory.
    """

    def __init__(self, out):
        try:
            os.makedirs(out)
        except OSError:
            pass
        self.file = open(os.path.join(out, 'log'), 'w')
        self.logs = []

    def write(self, log):
        """ Write log. """
        log = '[{}] - '.format(str(datetime.now())) + log
        tqdm.write(log)
        tqdm.write(log, file=self.file)
        self.file.flush()
        self.logs.append(log)

    def state_dict(self):
        """ Returns the state of the logger. """
        return {'logs': self.logs}

    def load_state_dict(self, state_dict):
        """ Loads the logger state. """
        self.logs = state_dict['logs']
        # write logs.
        tqdm.write(self.logs[-1])
        for log in self.logs:
            tqdm.write(log, file=self.file)


class BaseTrainer(object, metaclass=ABCMeta):

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def _test(self):
        pass

    @abstractmethod
    def fit(self):
        pass


class Trainer(BaseTrainer):
    """ Train pose net of estimating 2D pose from image.

    Args:
        data-augmentation (bool): Crop randomly and add random noise for data augmentation.
        epoch (int): Number of epochs to train.
        opt (str): Optimization method.
        gpu (bool): Use GPU.
        seed (str): Random seed to train.
        train (str): Path to training image-pose list file.
        val (str): Path to validation image-pose list file.
        batchsize (int): Learning minibatch size.
        out (str): Output directory.
        resume (str): Initialize the trainer from given file.
            The file name is 'epoch-{epoch number}.iter'.
        resume_model (str): Load model definition file to use for resuming training
            (it\'s necessary when you resume a training).
            The file name is 'epoch-{epoch number}.model'.
        resume_opt (str): Load optimization states from this file
            (it\'s necessary when you resume a training).
            The file name is 'epoch-{epoch number}.state'.
    """
    with open('config.yml', 'r') as f:
        configs = yaml.load(f)


    def __init__(self, **kwargs):
        self.data_augmentation = kwargs['data_augmentation']
        self.epoch = kwargs['epoch']
        self.gpu = (kwargs['gpu'] >= 0)
        self.opt = kwargs['opt']
        self.seed = kwargs['seed']
        self.train = kwargs['train']
        self.val = kwargs['val']
        self.batchsize = kwargs['batchsize']
        self.out = kwargs['out']
        self.resume = kwargs['resume']
        self.resume_model = kwargs['resume_model']
        self.resume_opt = kwargs['resume_opt']
        # validate arguments.
        self._validate_arguments()
        self.lowest_loss = None
        self.experiment = Experiment(api_key=self.configs['API_KEY'])
        self.experiment.log_multiple_params(kwargs)

    def _validate_arguments(self):
        if self.seed is not None and self.data_augmentation:
            raise NotSupportedError('It is not supported to fix random seed for data augmentation.')
        if self.gpu and not torch.cuda.is_available():
            raise GPUNotFoundError('GPU is not found.')
        #for path in (self.train, self.val):
        #    if not os.path.isfile(path):
        #        raise FileNotFoundError('{0} is not found.'.format(path))
        if self.opt not in ('MomentumSGD', 'Adam'):
            raise UnknownOptimizationMethodError(
                '{0} is unknown optimization method.'.format(self.opt))
        if self.resume is not None:
            for path in (self.resume, self.resume_model, self.resume_opt):
                if not os.path.isfile(path):
                    raise FileNotFoundError('{0} is not found.'.format(path))

    # TODO: make it acceptable multiple optimizer
    def _get_optimizer(self, model):
        if self.opt == 'MomentumSGD':
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif self.opt == "Adam":
            optimizer = optim.Adam(model.parameters())
        return optimizer

    def _train(self, model, optimizer, loss_func, train_iter, logger, start_time, log_interval=10):
        model.train()
        loss_sum = 0.0
        for iteration, batch in enumerate(tqdm(train_iter, desc='this epoch'), 1):
            data, target = Variable(batch[0]), Variable(batch[1])
            if self.gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss_sum += loss
            loss.backward()
            optimizer.step()
            self.experiment.set_step(iteration)
            self.experiment.log_metric("loss", loss.data[0])
            if iteration % log_interval == 0:
                log = 'elapsed_time: {0}, loss: {1}'.format(time.time() - start_time, loss.data[0])
                logger.write(log)
        return loss_sum / len(train_iter)

    def _test(self, model, test_iter, loss_func, logger, start_time):
        model.eval()
        test_loss = 0
        for batch in test_iter:
            data, target = Variable(batch[0]), Variable(batch[1])
            if self.gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += loss_func(output, target).data[0]
        test_loss /= len(test_iter)
        log = 'elapsed_time: {0}, validation/loss: {1}'.format(time.time() - start_time, test_loss)
        logger.write(log)
        return test_loss

    def _checkpoint(self, epoch, model, optimizer, logger):
        filename = os.path.join(self.out, 'pytorch', 'epoch-{0}'.format(epoch + 1))
        torch.save({'epoch': epoch + 1, 'logger': logger.state_dict()}, filename + '.iter')
        torch.save(model.state_dict(), filename + '.model')
        torch.save(optimizer.state_dict(), filename + '.state')

    def _best_checkpoint(self, epoch, model, optimizer, logger):
        filename = os.path.join(self.out, 'best_model')
        torch.save({'epoch': epoch + 1, 'logger': logger.state_dict()}, filename + '.iter')
        torch.save(model.state_dict(), filename + '.model')
        torch.save(optimizer.state_dict(), filename + '.state')

    def fit(self, model, train_data, val_data, loss_func):
        """ Train pose net. """
        # set random seed.
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.gpu:
                torch.cuda.manual_seed(self.seed)
        # initialize model to train.
        if self.resume_model:
            model.load_state_dict(torch.load(self.resume_model))
        # prepare gpu.
        if self.gpu:
            model.cuda()
        # load the datasets.
        input_transforms = [transforms.ToTensor()]
        # training/validation iterators.
        train_iter = torch.utils.data.DataLoader(train_data, batch_size=self.batchsize, shuffle=True)
        val_iter = torch.utils.data.DataLoader(val_data, batch_size=self.batchsize, shuffle=False)
        # set up an optimizer.
        optimizer = self._get_optimizer(model)
        if self.resume_opt:
            optimizer.load_state_dict(torch.load(self.resume_opt))
        # set intervals.
        val_interval = 10
        resume_interval = self.epoch/10
        log_interval = 10
        # set logger and start epoch.
        logger = TrainLogger(self.out)
        start_epoch = 0
        if self.resume:
            resume = torch.load(self.resume)
            start_epoch = resume['epoch']
            logger.load_state_dict(resume['logger'])
        # start training.
        start_time = time.time()
        loss = None
        for epoch in trange(start_epoch, self.epoch, initial=start_epoch, total=self.epoch, desc='     total'):
            self._train(model, optimizer, loss_func, train_iter, log_interval, logger, start_time)
            if (epoch + 1) % val_interval == 0:
                loss = self._test(model, val_iter, loss_func, logger, start_time)
                if self.lowest_loss == None or self.lowest_loss > loss:
                    logger.write('Best model updated. loss: {} => {}'.format(self.lowest_loss, loss))
                    self._best_checkpoint(epoch, model, optimizer, logger)
                    self.lowest_loss = loss
            if (epoch + 1) % resume_interval == 0:
                self._checkpoint(epoch, model, optimizer, logger)

    @staticmethod
    def get_args():
        # arg definition
        parser = argparse.ArgumentParser(
            description='Training pose net for comparison \
            between chainer and pytorch about implementing DeepPose.')
        parser.add_argument(
            '--data-augmentation', '-a', action='store_true', help='Crop randomly and add random noise for data augmentation.')
        parser.add_argument(
            '--epoch', '-e', type=int, default=100, help='Number of epochs to train.')
        parser.add_argument(
            '--opt', '-o', type=str, default='MomentumSGD',
            choices=['MomentumSGD', 'Adam'], help='Optimization method.')
        parser.add_argument(
            '--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU).')
        parser.add_argument(
            '--seed', '-s', type=int, help='Random seed to train.')
        parser.add_argument(
            '--train', type=str, default='data/train', help='Path to training image-pose list file.')
        parser.add_argument(
            '--val', type=str, default='data/test', help='Path to validation image-pose list file.')
        parser.add_argument(
            '--batchsize', type=int, default=32, help='Learning minibatch size.')
        parser.add_argument(
            '--out', default='result', help='Output directory')
        parser.add_argument(
            '--resume', default=None,
            help='Initialize the trainer from given file. \
            The file name is "epoch-{epoch number}.iter".')
        parser.add_argument(
            '--resume-model', type=str, default=None,
            help='Load model definition file to use for resuming training \
            (it\'s necessary when you resume a training). \
            The file name is "epoch-{epoch number}.mode"')
        parser.add_argument(
            '--resume-opt', type=str, default=None,
            help='Load optimization states from this file \
            (it\'s necessary when you resume a training). \
            The file name is "epoch-{epoch number}.state"')
        args = parser.parse_args()
        return args


if __name__ == '__main__':
    pass


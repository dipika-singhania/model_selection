from __future__ import print_function, division
import os
import sys
import torch
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from torch.utils.data.dataloader import default_collate
from utils import ValueMeter, topk_accuracy, topk_recall, softmax
import time
import copy
import json
from dataset_loaders.image_classification.data_config import DATA_DICTIONARY
from dataset_loaders.image_classification.utils import return_dataloader
from PIL import Image
from tqdm import tqdm
import sys

from model_bin.image_classification.models_pool import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


class DNN:
    def __init__(self, device='cpu', dataset_dict=None, model_name='resnet18', batch_size=128, max_epochs=20,
                 log_file_name=None, models_save_dir="model_pool/", optim_param={}, train_full=True):
        self.device = device
        self.layer_output_size = dataset_dict['num_classes']
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.model_zoo = ModelsZoo()
        self.ml_model = None
        self.model = self._build_model(model_name, dataset_dict['num_classes'], train_full)
        self.optimizer = self.create_optim('Adam', optim_param)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.criterion = self.create_criteria()
        self.loss_meter = ValueMeter()
        self.accuracy_meter = ValueMeter()
        #if log_file_name is not None:
        #    sys.stdout = open(os.path.join(log_file_name), 'w')
        self.best_model_wts = None
        self.best_perf = 0.0
        self.num_classes = dataset_dict['num_classes']
        self.start_epoch = 0
        self.best_epoch = 0     
        self.path_to_model = models_save_dir
        self.curr_epoch = 0
        self.dataset_dict = dataset_dict
        self.run_dict = self.update_initial_dict()
        self.final_logs = "logs/model_datasets_meta.json"

    def update_initial_dict(self):
        return {'model_name': self.model_name, 'model_params': self.get_params(),
                'dataset': self.dataset_dict['name'], 'num_classes': self.num_classes}

    def create_optim(self, name='Adam', params={'momentum': 0.8, 'lr': 0.001, 'beta_0': 0.9, 'beta_1': 0.999}):
        if name == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=params['lr'], momentum=params['momentum'])
        elif name == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=params['lr'], betas=(params['beta_0'], params['beta_1']))

    def create_criteria(self):
        return nn.CrossEntropyLoss()

    def log(self, mode, epoch, loss, accuracy, best_perf=None, green=False):
        if green:
            print('\033[92m', end="")

        print("[{}] Epoch: {:.2f}. ".format(mode, epoch),
                "Total Loss: {:.2f}. ".format(loss),
                "Accuracy: {:.2f}% ".format(accuracy),
                end="")

        if best_perf:
            print("[best: {:.2f}]%".format(best_perf), end="")

        print('\033[0m')

    def on_train_end(self, since):
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_perf))

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)
        # self.evaluate(self.dataset_dict['validation_loader'])
        self.evaluate(self.dataset_dict['test_loader'])

        self.run_dict['best_epoch'] = self.best_epoch
        self.run_dict['total_train_time'] = time_elapsed
        self.run_dict['total_epoch_run'] = self.max_epochs
        self.run_dict['train_time_per_image'] = time_elapsed / self.run_dict['train_data_size']

    def get_dict(self):
        return self.run_dict

    def on_train_epoch_end(self, epoch):
        # log at the end of each epoch
        self.log('train', epoch + 1, self.loss_meter.value(), self.accuracy_meter.value(), None, green=True)
        # deep copy the model
        self.evaluate(self.dataset_dict['validation_loader'])
        self.dump_parameters()

    def on_train_iter_end(self, outputs, labels, loss, bs, e):
        train_acc = topk_accuracy(outputs, labels, (1,))[0] * 100
        self.loss_meter.add(loss, bs)
        self.accuracy_meter.add(train_acc, bs)
        self.log('train', e, self.loss_meter.value(), self.accuracy_meter.value())

    def train_ml_model(self):
        since = time.time()
        #import pdb
        #pdb.set_trace()
        # Each epoch has a training and validation phase
        self.model.eval()  # Set model to training mode
        self.curr_epoch = 1
        data_size = 0
        # Iterate over data.
        final_input = []
        final_output = []
        for i, (inputs, labels) in enumerate(self.dataset_dict['train_loader']):
            inputs = inputs.to(self.device)
            labels = labels.long()
            labels = labels.to(self.device)
            bs = labels.shape[0]  # batch size
            data_size += bs
            outputs = self.model(inputs)
            final_input.append(outputs)
            final_output.append(labels)

        #import pdb
        #pdb.set_trace()
        final_input = torch.concat(final_input, 0)
        final_output = torch.concat(final_output, 0)

        # pca = PCA(n_components=200)
        # pca.fit(final_input)
        # final_input = pca.fit_transform(final_input)

        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(final_input, final_output)          
        self.ml_model = clf

        predict_train = clf.predict(final_input)
        train_acc = accuracy_score(predict_train, final_output)
        self.log('train', 0, 0, train_acc)

    def eval_ml_model(self):
        
        #for i, (inputs, labels) in enumerate(validation_loader):
        return
        
    def train(self):
        if self.ml_model is not None:
            self.train_ml_model()
            return
        since = time.time()

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        data_size = 0

        for epoch in range(self.start_epoch, self.max_epochs):
            # Each epoch has a training and validation phase
            self.model.train()  # Set model to training mode
            self.curr_epoch = epoch
            # Iterate over data.
            for i, (inputs, labels) in enumerate(self.dataset_dict['train_loader']):
                inputs = inputs.to(self.device)
                labels = labels.long()
                labels = labels.to(self.device)
                bs = labels.shape[0]  # batch size
                data_size += bs

                # zero the parameter gradients
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                e = epoch + (i/len(self.dataset_dict['train_loader']))
                self.on_train_iter_end(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                                       loss.detach().item(), bs, e)

            self.on_train_epoch_end(epoch)
    
        self.run_dict['train_data_size'] = data_size
        self.on_train_end(since)

    def get_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def on_eval_end(self, predictions, act_labels, loss, start_time):
        predictions = np.concatenate(predictions)
        act_labels = np.concatenate(act_labels)
        action_accuracies_1 = topk_accuracy(predictions, act_labels, (1,))[0] * 100
        action_accuracies_5 = topk_accuracy(predictions, act_labels, (5,))[0] * 100
        action_recall_1 = topk_recall(predictions, act_labels, 1) * 100
        action_recall_5 = topk_recall(predictions, act_labels, 5) * 100

        if action_accuracies_1 > self.best_perf:
            self.best_perf = action_accuracies_1
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.curr_epoch

        self.log('eval', self.curr_epoch + 1, loss, action_accuracies_1, self.best_perf, green=True)

        total_val_time = time.time() - start_time

        self.run_dict['top_5_acc'] = action_accuracies_5
        self.run_dict['top_5_recall'] = action_recall_5
        self.run_dict['top_1_recall'] = action_recall_1
        self.run_dict['top_1_acc'] = action_accuracies_1
        self.run_dict['eval_time'] = total_val_time
        self.run_dict['eval_time_per_image'] = total_val_time / self.run_dict['eval_data_size']
        
    def evaluate(self, loader):
        since = time.time()
        self.model.eval()
        predictions = []
        act_labels = []
        loss = 0
        eval_data_size = 0
        with torch.set_grad_enabled(False):
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.long()
                labels = labels.to(self.device)
                eval_data_size += len(labels)
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                predictions.append(output.cpu().numpy())
                act_labels.append(labels.cpu().numpy())

        self.run_dict['eval_data_size'] = eval_data_size
        self.on_eval_end(predictions, act_labels, loss, since)

    def get_model_dump_name(self):
        return self.model_name + "_" + str(self.dataset_dict['name']) + ".pth.tar"

    def dump_parameters(self):
        exp_name = self.get_model_dump_name()
        torch.save({'state_dict': self.best_model_wts, 'epoch': self.best_epoch, 
                    'best_perf':self.best_perf, 'optim':self.optimizer.state_dict()}, 
                   os.path.join(self.path_to_model, exp_name))
        return

    def load_parameters(self):
        # Load model parameters
        exp_name = self.get_model_dump_name()
        chk = torch.load(os.path.join(self.path_to_model, exp_name))
        self.model.load_state_dict(chk['state_dict'])
        self.best_perf = chk['best_perf']
        self.best_epoch = chk['epoch']
        self.optimizer.load_state_dict(chk['optim'])
        return

    def resume_train(self):
        self.load_parameters()
        self.start_epoch = self.best_epoch
        self.model.to(self.device)
        self.model.train()
        self.train()

    def _build_model(self, model_name, num_classes, train_full=True):
        if model_name in ['DecisionTreeClassifier', 'SVM', 'AdaBoostClassifier']:
            model = self.model_zoo.extract_resnet50_features()
            model = model.to(self.device)
            self.ml_model = model_name
            return model

        model = self.model_zoo.get_modified_model(model_name, num_classes, train_full)
        model = model.to(self.device)
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_name', type=str, 
                        help='Path(s) to query image(s), delimited by commas')
    parser.add_argument('--final_log', type=str, help="Final log storage")
    parser.add_argument('--model_save_dir', type=str, help="Model save dir", default="model_pool/")
    parser.add_argument('--model_name', type=str, 
                        choices=['resnet18','shufflenet_v2_x0_5', 'resnext50_32x4d', 'mnasnet0_5', 'mnasnet1_0',
                                 'mnasnet1_3', 'resnet34', 'resnext50', 'resnet50', 'DecisionTreeClassifier'],
                        help="Pytorch model name to run", default="resnet18")
    parser.add_argument('--batch_size', type=float, default=64, help="batch_size of model")
    parser.add_argument('--max_epoch', type=int, default=100, help="max epochs to run a model")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate to run a model")
    parser.add_argument('--train_full', action='store_true', help="Trains the full network")
    parser.add_argument('--dataset', choices=list(DATA_DICTIONARY.keys()), help="Name of the dataset")
    parser.add_argument('--data_dir', default='dataset_pool/', help="Data location")

    (args, _) = parser.parse_known_args()

    dataset_dict = return_dataloader(root='dataset_pool/', batch_size=args.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optim_param = {'momentum': 0.8, 'lr': args.lr, 'beta_0': 0.9, 'beta_1': 0.999}
    dnn = DNN(device, model_name=args.model_name, batch_size=args.batch_size, max_epochs=args.max_epoch,
              dataset_dict=dataset_dict, log_file_name=args.log_file_name, models_save_dir=args.model_save_dir,
              optim_param=optim_param, train_full=args.train_full)
    dnn.train()
    dict_params = dnn.get_dict()
    dict_params['dataset'] = args.dataset

    with open(args.final_log, 'a+') as fp:
        json_string = json.dumps(dict_params)
        fp.write(json_string + "\n")

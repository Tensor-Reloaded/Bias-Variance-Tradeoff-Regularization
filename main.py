import collections
import sys
import pprint
import argparse
import pickle
import os
from shutil import copyfile

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from tensorboardX import SummaryWriter
from torchvision import transforms as transforms
import hydra
from hydra import utils
from omegaconf import DictConfig

from learn_utils import *
from misc import progress_bar
from models import *

APEX_MISSING = False
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("Apex not found on the system, it won't be using half-precision")
    APEX_MISSING = True
    pass


storage_dir = "../storage/"

@hydra.main(config_path='experiments/config.yaml', strict=True)
def main(config: DictConfig):
    global storage_dir
    storage_dir = os.path.dirname(utils.get_original_cwd()) + "/storage/"
    save_config_path = "runs/" + config.save_dir
    os.makedirs(save_config_path, exist_ok=True)
    with open(os.path.join(save_config_path, "README.md"), 'w+') as f:
        f.write(config.pretty())

    if APEX_MISSING:
        config.half = False

    solver = Solver(config)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.args = config
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.es = EarlyStopping(patience=self.args.es_patience)
        if not self.args.save_dir:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir="runs/" + self.args.save_dir)

        if self.args.homomorphic_regularization:
            self.t = 1.0
            self.n = self.args.homomorphic_k_inputs
            self.k = self.n-1
            self.centroid = 1/(self.n-self.k) - self.k/((self.n-self.k)*(self.n-self.k-1)) + (self.t*(self.n-1))/((self.n-self.k)*(self.n-self.k-1))
            self.remainder = 1 - (self.centroid * (self.n-self.k-1))
            self.sum_groups = self.n - self.k

        self.batch_plot_idx = 0

        self.train_batch_plot_idx = 0
        self.test_batch_plot_idx = 0
        self.val_batch_plot_idx = 0
        if self.args.dataset == "CIFAR-10":
            self.nr_classes = len(CIFAR_10_CLASSES)
        elif self.args.dataset == "CIFAR-100":
            self.nr_classes = len(CIFAR_100_CLASSES)

    def load_data(self):
        if "CIFAR" in self.args.dataset:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(
            ), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), normalize])
        else:
            train_transform = transforms.Compose([transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.ToTensor()])

        if self.args.dataset == "CIFAR-10":
            self.train_set = torchvision.datasets.CIFAR10(
                root=storage_dir, train=True, download=True, transform=train_transform)
        elif self.args.dataset == "CIFAR-100":
            self.train_set = torchvision.datasets.CIFAR100(
                root=storage_dir, train=True, download=True, transform=train_transform)

        if self.args.train_subset is None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_set, batch_size=self.args.train_batch_size, shuffle=True)
        else:
            filename = "subset_indices/subset_balanced_{}_{}.data".format(
                self.dataset, self.args.train_subset)
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    subset_indices = pickle.load(f)
            else:
                subset_indices = []
                per_class = self.args.train_subset // self.nr_classes
                targets = torch.tensor(self.train_set.targets)
                for i in range(self.nr_classes):
                    idx = (targets == i).nonzero().view(-1)
                    perm = torch.randperm(idx.size(0))[:per_class]
                    subset_indices += idx[perm].tolist()
                if not os.path.isdir("subset_indices"):
                    os.makedirs("subset_indices")
                with open(filename, 'wb') as f:
                    pickle.dump(subset_indices, f)
            subset_indices = torch.LongTensor(subset_indices)
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_set, batch_size=self.args.train_batch_size,
                sampler=SubsetRandomSampler(subset_indices))
            if self.args.validate:
                self.validate_loader = torch.utils.data.DataLoader(
                    dataset=self.train_set, batch_size=self.args.train_batch_size,
                    sampler=SubsetRandomSampler(subset_indices))

        if self.args.dataset == "CIFAR-10":
            test_set = torchvision.datasets.CIFAR10(
                root=storage_dir, train=False, download=True, transform=test_transform)
        elif self.args.dataset == "CIFAR-100":
            test_set = torchvision.datasets.CIFAR100(
                root=storage_dir, train=False, download=True, transform=test_transform)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda' + ":" + str(self.args.cuda_device))
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = eval(self.args.model)
        self.save_dir = storage_dir + self.args.save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.init_model()
        if len(self.args.load_model) > 0:
            print("Loading model from " + self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model))
        self.model = self.model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        print(self.args.scheduler_name)
        if self.args.scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.args.lr_gamma, patience=self.args.reduce_lr_patience,
                min_lr=self.args.reduce_lr_min_lr, verbose=True, threshold=self.args.reduce_lr_delta)
        elif self.args.scheduler == "CosineAnnealingLR":
            if self.args.sum_augmentation:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.args.epoch//(self.args.nr_cycle-1),eta_min=self.args.reduce_lr_min_lr)
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.args.epoch,eta_min=self.args.reduce_lr_min_lr)
        elif self.args.scheduler == "MultiStepLR":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_milestones, gamma=self.args.lr_gamma)
        elif self.args.scheduler == "OneCycleLR":
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.args.lr, total_steps=None, epochs=self.args.epoch//(self.args.nr_cycle-1), steps_per_epoch=len(self.train_loader), pct_start=self.args.pct_start, anneal_strategy=self.args.anneal_strategy, cycle_momentum=self.args.cycle_momentum, base_momentum=self.args.base_momentum, max_momentum=self.args.max_momentum, div_factor=self.args.div_factor, final_div_factor=self.args.final_div_factor, last_epoch=self.args.last_epoch)
        else:
            print("This scheduler is not implemented, go ahead an commit one")

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if self.cuda:
            if self.args.half:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=f"O{self.args.mixpo}",
                                                            patch_torch_functions=True, keep_batchnorm_fp32=True)

    def get_batch_plot_idx(self):
        self.batch_plot_idx += 1
        return self.batch_plot_idx - 1

    def forward_lipschitz_loss_hook_fn(self,module,X,y):
        if not self.model.training  or not self.args.lipschitz_regularization or (self.args.level == "layer" and not hasattr(module,'weight')):
            return
        module.eval()
        
        module.forward_handle.remove()

        X = X[0]
        y = module(X)
        noise = self.args.lipschitz_noise_factor * torch.std(X, dim=0) * torch.randn(X.size(), device=self.device) 
        X = X + noise
        X = module(X)

        if self.args.distance_function == "cosine_loss":
            if self.lipschitz_loss is None:
                self.lipschitz_loss = F.cosine_embedding_loss(X,y,self.aux_y)
            else:
                self.lipschitz_loss += F.cosine_embedding_loss(X,y,self.aux_y)
        elif self.args.distance_function == "mse":
            if self.lipschitz_loss is None:
                self.lipschitz_loss = F.mse_loss(X,y)
            else:
                self.lipschitz_loss += F.mse_loss(X,y)
        elif self.args.distance_function == "nll":
            if self.lipschitz_loss is None:
                self.lipschitz_loss =  (-F.softmax(y)+F.softmax(X).exp().sum(0).log()).mean()
            else:
                self.lipschitz_loss += (-F.softmax(y)+F.softmax(X).exp().sum(0).log()).mean()
        else:
            print("lipschitz distance function not implemented")
            exit()
        
        module.forward_handle = module.register_forward_hook(self.forward_lipschitz_loss_hook_fn)

    def forward_homomorphic_loss_hook_fn(self,module,X,y):
        if not self.model.training  or not self.args.homomorphic_regularization:
            return

        module.forward_handle.remove()

        X = X[0]
        shuffled_idxs = torch.randperm(y.size(0), device=self.device, dtype=torch.long)
        shuffled_idxs = shuffled_idxs[:y.size(0)-y.size(0) % self.sum_groups]
        mini_batches_idxs = shuffled_idxs.split(y.size(0) // self.sum_groups)

        to_sum_groups = []
        to_sum_targets = []
        for mbi in mini_batches_idxs:
            to_sum_groups.append(X[mbi].unsqueeze(0))
            to_sum_targets.append(y[mbi].unsqueeze(0))

        k_weights = torch.full((1,self.n),1/self.n)
        if self.args.gradual_cascade:
            k_weights = self.get_k_weights()
        k_weights = k_weights.to(self.device)
        data = (torch.cat(to_sum_groups, dim=0).T*k_weights[:,:self.sum_groups]).T.sum(0)
        data = module(data)
        targets = (torch.cat(to_sum_targets, dim=0).T*k_weights[:,:self.sum_groups]).T.sum(0)

        if self.args.distance_function == "cosine_loss":
            if self.homomorphic_loss is None:
                self.homomorphic_loss = F.cosine_embedding_loss(data,targets,self.aux_y)
            else:
                self.homomorphic_loss.add(F.cosine_embedding_loss(data,targets,self.aux_y))
        elif self.args.distance_function == "mse":
            if self.homomorphic_loss is None:
                self.homomorphic_loss = F.mse_loss(data,targets)
            else:
                self.homomorphic_loss += F.mse_loss(data,targets)
        elif self.args.distance_function == "nll":
            if self.homomorphic_loss is None:
                self.homomorphic_loss =  (-F.softmax(targets)+F.softmax(data).exp().sum(0).log()).mean()
            else:
                self.homomorphic_loss += (-F.softmax(targets)+F.softmax(data).exp().sum(0).log()).mean()
        else:
            print("Homomorphic distance function not implemented")
            exit()
        
        module.forward_handle = module.register_forward_hook(self.forward_homomorphic_loss_hook_fn)

    def add_lipschitz_regularization(self):
        self.modules_count = 0
        self.aux_y = torch.ones((1), device=self.device)
        if self.args.level == "model":
            self.modules_count = 1
            self.model.forward_handle = self.model.register_forward_hook(self.forward_lipschitz_loss_hook_fn)

        elif self.args.level == "block":
            if "PreResNet" in self.args.model_name:
                for name, module in self.model.named_modules():
                    if re.match(r"^layer[0-9]\.[0-9]+$", name):
                        self.modules_count += 1
                        module.forward_handle = module.register_forward_hook(self.forward_lipschitz_loss_hook_fn)

        elif self.args.level == "layer":
            modules = []
            def remove_sequential(network, modules):
                for layer in network.children():
                    if len(list(layer.children())) > 0:
                        remove_sequential(layer,modules)
                    if len(list(layer.children())) == 0:
                        modules.append(layer)
            remove_sequential(self.model,modules)


            for i,module in enumerate(modules):
                self.modules_count += 1
                module.forward_handle = module.register_forward_hook(self.forward_lipschitz_loss_hook_fn)

    def add_homomorphic_regularization(self):
        self.modules_count = 0
        self.aux_y = torch.ones((1), device=self.device)
        if self.args.level == "model":
            self.modules_count = 1
            self.model.forward_handle = self.model.register_forward_hook(self.forward_homomorphic_loss_hook_fn)

        elif self.args.level == "block":
            if "PreResNet" in self.args.model_name:
                for name, module in self.model.named_modules():
                    if re.match(r"^layer[0-9]\.[0-9]+$", name):
                        module.forward_handle = module.register_forward_hook(self.forward_homomorphic_loss_hook_fn)

        elif self.args.level == "layer":
            modules = []
            def remove_sequential(network, modules):
                for layer in network.children():
                    if len(list(layer.children())) > 0:
                        remove_sequential(layer,modules)
                    if len(list(layer.children())) == 0:
                        modules.append(layer)
            remove_sequential(self.model,modules)


            for i,module in enumerate(modules):
                self.modules_count += 1
                module.forward_handle = module.register_forward_hook(self.forward_homomorphic_loss_hook_fn)

    def get_k_weights(self):
        eps = self.remainder * (self.t-(self.k/(self.n-1)))/(self.n-1)

        weights = torch.zeros(self.n)
        weights[:self.n-self.k-1] = self.centroid + eps/(self.n-self.k-1)
        weights[self.n-self.k-1] = self.remainder - eps
        weights[self.n-self.k:] = 0.0

        return weights.unsqueeze(0)
    
    def train(self):
        print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            batch_idx = self.get_batch_plot_idx()

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            
            if self.args.lipschitz_regularization:
                self.lipschitz_loss = (self.lipschitz_loss * self.args.lipschitz_regularization_loss_factor)/self.modules_count
                loss += self.lipschitz_loss
                self.writer.add_scalar("Train/Lipschitz Batch_Loss", self.lipschitz_loss.item(), batch_idx) # TODO the loss values suck, they are either ~1.0 or ~0.0

            if self.args.homomorphic_regularization:
                self.homomorphic_loss = (self.homomorphic_loss * self.args.homomorphic_regularization_factor)/self.modules_count
                loss += self.homomorphic_loss
                self.writer.add_scalar("Train/Homomorphic Batch_Loss", self.homomorphic_loss.item(), batch_idx)

            self.writer.add_scalar("Train/Batch_Loss", loss.item(), batch_idx)
            
            if self.args.half:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            total_loss += loss.item()


            prediction = torch.max(output, 1)
            total += target.size(0)

            correct += torch.sum((prediction[1] == target).float()).item()

            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (total_loss / (batch_num + 1), 100.0 * correct/total, correct, total))
            if self.args.scheduler == "OneCycleLR":
                self.scheduler.step()
        return total_loss, correct / total

    def test(self):
        print("test:")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.add_scalar("Test/Batch_Loss", loss.item(), self.get_batch_plot_idx())
                total_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += torch.sum((prediction[1] == target).float()).item()

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return total_loss, correct/total

    def save(self, epoch, accuracy, tag=None):
        if tag != None:
            tag = "_"+tag
        else:
            tag = ""
        model_out_path = self.save_dir + \
            "/model_{}_{}{}.pth".format(
                epoch, accuracy * 100, tag)
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        if self.args.seed is not None:
            reset_seed(self.args.seed)
        self.load_data()
        self.load_model()

        if self.args.lipschitz_regularization:
            self.add_lipschitz_regularization()

        if self.args.homomorphic_regularization:
            self.add_homomorphic_regularization()

        best_accuracy = 0
        try:
            for epoch in range(1, self.args.epoch + 1):
                if self.args.lipschitz_regularization and epoch in self.args.lipschitz_noise_factor_milestines:
                    self.args.lipschitz_noise_factor *= self.args.lipschitz_noise_factor_gamma
                if self.args.homomorphic_regularization and epoch in self.args.homomorphic_k_hot_milestines:
                    self.t -= (1.0/self.args.homomorphic_k_inputs) * self.args.homomorphic_k_hot_gamma 
                print("\n===> epoch: %d/%d" % (epoch, self.args.epoch))

                train_result = self.train()

                loss = train_result[0]
                accuracy = train_result[1]
                self.writer.add_scalar("Train/Loss", loss, epoch)
                self.writer.add_scalar("Train/Accuracy", accuracy, epoch)

                test_result = self.test()

                loss = test_result[0]
                accuracy = test_result[1]
                self.writer.add_scalar("Test/Loss", loss, epoch)
                self.writer.add_scalar("Test/Accuracy", accuracy, epoch)

                self.writer.add_scalar("Model/Norm", self.get_model_norm(), epoch)
                self.writer.add_scalar("Train_Params/Learning_rate", self.scheduler.get_last_lr()[0], epoch)
                if self.args.lipschitz_regularization:
                    self.writer.add_scalar("Train_Params/Lipschitz_noise_factor", self.args.lipschitz_noise_factor, epoch)
                if self.args.homomorphic_regularization:
                    self.centroid = 1/(self.n-self.k) - self.k/((self.n-self.k)*(self.n-self.k-1)) + (self.t*(self.n-1))/((self.n-self.k)*(self.n-self.k-1))
                    self.remainder = 1 - (self.centroid * (self.n-self.k-1))
                    self.writer.add_scalar("Train_Params/Homomorphic_K-hot", self.n-self.t * (self.n - 1), epoch)

                if best_accuracy < test_result[1]:
                    best_accuracy = test_result[1]
                    self.save(epoch, best_accuracy)
                    print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))

                if self.args.save_model and epoch % self.args.save_interval == 0:
                    self.save(epoch, 0)

                if self.args.scheduler == "MultiStepLR":
                    self.scheduler.step()
                elif self.args.scheduler == "ReduceLROnPlateau":
                    self.scheduler.step(train_result[0])
                elif self.args.scheduler == "OneCycleLR":
                    pass
                else:
                    self.scheduler.step()

                if self.es.step(train_result[0]):
                    print("Early stopping")
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))
        files = os.listdir(self.save_dir)
        paths = [os.path.join(self.save_dir, basename) for basename in files if "_0" not in basename]
        if len(paths) > 0:
            src = max(paths, key=os.path.getctime)
            copyfile(src, os.path.join("runs", self.args.save_dir, os.path.basename(src)))

        with open("runs/" + self.args.save_dir + "/README.md", 'a+') as f:
            f.write("\n## Accuracy\n %.3f%%" % (best_accuracy * 100))
        print("Saved best accuracy checkpoint")

    def get_model_norm(self, norm_type=2):
        norm = 0.0
        for param in self.model.parameters():
            norm += torch.norm(input=param, p=norm_type, dtype=torch.float)
        return norm

    def init_model(self):
        if self.args.initialization == 1:
            # xavier init
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform(
                        m.weight, gain=nn.init.calculate_gain('relu'))
        elif self.args.initialization == 2:
            # he initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal(m.weight, mode='fan_in')
        elif self.args.initialization == 3:
            # selu init
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.kernel_size[0] * \
                        m.kernel_size[1] * m.in_channels
                    nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
        elif self.args.initialization == 4:
            # orthogonal initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal(m.weight)

        if self.args.initialization_batch_norm:
            # batch norm initialization
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant(m.weight, 1)
                    nn.init.constant(m.bias, 0)


if __name__ == '__main__':
    main()

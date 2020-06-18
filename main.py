import collections
import sys
import pprint
import argparse
import pickle
import os
import re
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


def nll_loss(got, want):
    return (-F.softmax(want)+F.softmax(got).exp().sum(0).log()).mean()


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
            log_dir = "runs/" + self.args.save_dir
            log_dir = os.path.abspath(log_dir)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f'Started tensorboardX.SummaryWriter(log_dir={log_dir})')

        if self.args.homomorphic_regularization:
            self.t = 1.0
            self.n = self.args.homomorphic_k_inputs
            self.k = self.n-1
            self.centroid = 1.0 #1/(self.n-self.k) - self.k/((self.n-self.k)*(self.n-self.k-1)) + (self.t*(self.n-1))/((self.n-self.k)*(self.n-self.k-1))
            self.remainder = 0.0 #1 - (self.centroid * (self.n-self.k-1))
            self.sum_groups = 1 #self.n - self.k

        self.batch_plot_idx = 0

        self.train_batch_plot_idx = 0
        self.test_batch_plot_idx = 0
        self.val_batch_plot_idx = 0
        if self.args.dataset == "CIFAR-10":
            self.nr_classes = len(CIFAR_10_CLASSES)
        elif self.args.dataset == "CIFAR-100":
            self.nr_classes = len(CIFAR_100_CLASSES)

        self.lipschitz_loss = None
        self.homomorphic_loss = None
        self.lipschitz_modules_count = None
        self.homomorphic_modules_count = None
        self.COSINE_EMBEDDING_LOSS_TARGET = None

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

        pin_memory = self.args.cuda

        if self.args.dataset == "CIFAR-10":
            self.train_set = torchvision.datasets.CIFAR10(
                root=storage_dir, train=True, download=True, transform=train_transform)
        elif self.args.dataset == "CIFAR-100":
            self.train_set = torchvision.datasets.CIFAR100(
                root=storage_dir, train=True, download=True, transform=train_transform)

        if self.args.train_subset is None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_set, batch_size=self.args.train_batch_size, shuffle=True, pin_memory=pin_memory)
        else:
            filename = "subset_indices/subset_balanced_{}_{}.data".format(
                self.args.dataset, self.args.train_subset)
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

        if self.args.dataset == "CIFAR-10":
            test_set = torchvision.datasets.CIFAR10(
                root=storage_dir, train=False, download=True, transform=test_transform)
        elif self.args.dataset == "CIFAR-100":
            test_set = torchvision.datasets.CIFAR100(
                root=storage_dir, train=False, download=True, transform=test_transform)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False, pin_memory=pin_memory)

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

        self.COSINE_EMBEDDING_LOSS_TARGET = torch.ones(1, device=self.device)  # TODO: ones(1, ...) or ones((1,), ...) ?

    def get_batch_plot_idx(self):
        self.batch_plot_idx += 1
        return self.batch_plot_idx - 1

    def compute_lips_homo_loss(self, got, want):
        if self.args.distance_function == "cosine_loss":
            return F.cosine_embedding_loss(got, want, self.COSINE_EMBEDDING_LOSS_TARGET,margin=0.0)
        elif self.args.distance_function == "mse":
            return F.mse_loss(got, want)
        elif self.args.distance_function == "nll":
            return nll_loss(got, want)
        elif self.args.distance_function == "bce_with_logits":
            return F.binary_cross_entropy_with_logits(got, want)
        elif self.args.distance_function == "cross_entropy":
            return F.cross_entropy(got, want)

        raise ValueError("lipschitz/homomorphic distance function not implemented")

    def forward_lipschitz_loss_hook_fn(self,module,X,y):
        if not self.model.training or not self.args.lipschitz_regularization or module.hook_in_progress:
            return
        module.hook_in_progress = True
        # module.eval()

        X = X[0].detach()
        y = y.detach()
        # noise = torch.randn(X.size(), device=self.device) * self.args.lipschitz_noise_factor
        noise = torch.randn(X.size(), device=self.device) * torch.std(X, dim=0) * self.args.lipschitz_noise_factor
        X = X + noise
        X = module(X)

        self.lipschitz_loss += self.compute_lips_homo_loss(X, y)

        # module.train()
        module.hook_in_progress = False

    def forward_homomorphic_loss_hook_fn(self,module,X,y):
        if not self.model.training or not self.args.homomorphic_regularization or module.hook_in_progress or self.sum_groups == 1:
            return
        module.hook_in_progress = True
        # module.eval()

        X = X[0].detach()
        y = y.detach()
        shuffled_idxs = torch.randperm(y.size(0), device=self.device, dtype=torch.long)
        shuffled_idxs = shuffled_idxs[:y.size(0)-y.size(0) % self.sum_groups]
        mini_batches_idxs = shuffled_idxs.split(y.size(0) // self.sum_groups)

        to_sum_groups = []
        to_sum_targets = []
        for mbi in mini_batches_idxs:
            to_sum_groups.append(X[mbi].unsqueeze(0))
            to_sum_targets.append(y[mbi].unsqueeze(0))

        assert self.sum_groups > 1

        k_weights = self.get_k_weights()

        data = (torch.cat(to_sum_groups, dim=0).T*k_weights[:,:self.sum_groups]).T.sum(0)
        data = module(data)
        targets = (torch.cat(to_sum_targets, dim=0).T*k_weights[:,:self.sum_groups]).T.sum(0)

        self.homomorphic_loss += self.compute_lips_homo_loss(data, targets)

        # module.train()
        module.hook_in_progress = False

    def add_regularization_forward_hook(self, level, handle_name, hook):
        modules_to_hook = []

        if level == "model":
            modules_to_hook.append(self.model)
        elif level == "superblock":
            for module in self.model.children():
                if hasattr(module, 'custom_name') and module.custom_name == 'SuperBlock':
                    modules_to_hook.append(module)
        elif level == "block":
            assert "PreResNet" in self.args.model_name
            for name, module in self.model.named_modules():
                if re.match(r"^layer[0-9]\.[0-9]+$", name):
                    modules_to_hook.append(module)
        elif level == "layer":
            def get_leaf_modules(network):
                leafs = []
                for layer in network.children():
                    is_leaf = len(list(layer.children())) == 0
                    if is_leaf:
                        leafs.append(layer)
                    else:
                        leafs.extend(get_leaf_modules(layer))
                return leafs

            # This assumes that self.model is not a leaf module!
            leaf_modules = get_leaf_modules(self.model)

            for i, module in enumerate(leaf_modules):
                if not hasattr(module, 'weight'):
                    continue
                modules_to_hook.append(module)
        else:
            raise ValueError('Unknown level ' + level)

        for module in modules_to_hook:
            handle = module.register_forward_hook(hook)
            setattr(module, handle_name, handle)
            module.hook_in_progress = False

        print('modules count:', len(modules_to_hook))
        return len(modules_to_hook)

    def add_lipschitz_regularization(self):
        self.lipschitz_modules_count = self.add_regularization_forward_hook(self.args.lipschitz_level, 'lipschitz_handle', self.forward_lipschitz_loss_hook_fn)

    def add_homomorphic_regularization(self):
        self.homomorphic_modules_count = self.add_regularization_forward_hook(self.args.homomorphic_level, 'homomorphic_handle', self.forward_homomorphic_loss_hook_fn)

    def get_k_weights(self):
        if self.args.homomorphic_const_sum_groups:
            return torch.empty(self.sum_groups, device=self.device).fill_(1.0 / self.sum_groups).unsqueeze(0)

        t = self.t
        n = self.n
        k = self.k

        eps = self.remainder * (t - (k / (n - 1))) / (n - 1)

        weights = torch.zeros(self.sum_groups, device=self.device)
        weights[:n - k - 1] = self.centroid + eps / (n - k - 1)
        weights[n - k - 1] = self.remainder - eps
        weights[n - k:] = 0.0

        return weights.unsqueeze(0)
    
    def train(self):
        print("train:")

        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            batch_plot_idx = self.get_batch_plot_idx()

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            self.model.train()
            self.lipschitz_loss = 0.0
            self.homomorphic_loss = 0.0
            output = self.model(data)
            loss = self.criterion(output, target)

            if self.args.lipschitz_regularization:
                self.lipschitz_loss = (self.lipschitz_loss * self.args.lipschitz_regularization_loss_factor) / self.lipschitz_modules_count
                loss += self.lipschitz_loss
                self.writer.add_scalar("Train/Lipschitz_Batch_Loss", self.lipschitz_loss.item(), batch_plot_idx) # TODO the loss values suck, they are either ~1.0 or ~0.0

            if self.args.homomorphic_regularization and self.sum_groups > 1:
                self.homomorphic_loss = (self.homomorphic_loss * self.args.homomorphic_regularization_factor) / self.homomorphic_modules_count
                loss += self.homomorphic_loss
                self.writer.add_scalar("Train/Homomorphic_Batch_Loss", self.homomorphic_loss.item(), batch_plot_idx)

            self.writer.add_scalar("Train/Batch_Loss", loss.item(), batch_plot_idx)
            
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
        if tag is not None:
            tag = "_" + tag
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
                if self.args.homomorphic_regularization and \
                        not self.args.homomorphic_const_sum_groups and \
                        epoch in self.args.homomorphic_k_hot_milestines:
                    self.t -= (1.0/self.args.homomorphic_k_inputs) * self.args.homomorphic_k_hot_gamma 
                    self.k = int(np.floor(self.t * (self.n - 1)))
                    n = self.n
                    k = self.k
                    self.sum_groups = n - k
                    self.centroid = 1 / (n - k) - k / ((n - k) * (n - k - 1)) + (self.t * (n - 1)) / ((n - k) * (n - k - 1))
                    self.remainder = 1 - (self.centroid * (n - k - 1))
                elif self.args.homomorphic_regularization and self.args.homomorphic_const_sum_groups:
                    self.sum_groups = self.args.homomorphic_k_inputs

                if self.args.scheduler in ["OneCycleLR"] and epoch % (self.args.epoch//(self.args.nr_cycle-1)) == 1:
                    self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.args.lr, total_steps=None, epochs=self.args.epoch//(self.args.nr_cycle-1), steps_per_epoch=len(self.train_loader), pct_start=self.args.pct_start, anneal_strategy=self.args.anneal_strategy, cycle_momentum=self.args.cycle_momentum, base_momentum=self.args.base_momentum, max_momentum=self.args.max_momentum, div_factor=self.args.div_factor, final_div_factor=self.args.final_div_factor, last_epoch=self.args.last_epoch)

                print("\n===> epoch: %d/%d" % (epoch, self.args.epoch))

                with Timer('epoch_train', verbose=False):
                    train_result = self.train()

                loss = train_result[0]
                accuracy = train_result[1]
                self.writer.add_scalar("Train/Loss", loss, epoch)
                self.writer.add_scalar("Train/Accuracy", accuracy, epoch)

                with Timer('epoch_test', verbose=False):
                    test_result = self.test()

                loss = test_result[0]
                accuracy = test_result[1]
                self.writer.add_scalar("Test/Loss", loss, epoch)
                self.writer.add_scalar("Test/Accuracy", accuracy, epoch)

                self.writer.add_scalar("Model/Norm", self.get_model_norm(), epoch)
                self.writer.add_scalar("Train_Params/Learning_rate", self.scheduler.get_last_lr()[0], epoch)
                if self.args.lipschitz_regularization:
                    self.writer.add_scalar("Train_Params/Lipschitz_noise_factor", self.args.lipschitz_noise_factor, epoch)

                if self.args.homomorphic_regularization and not self.args.homomorphic_const_sum_groups:
                    self.writer.add_scalar("Train_Params/Homomorphic_K-hot", self.n-self.t * (self.n - 1), epoch)
                elif self.args.homomorphic_regularization and self.args.homomorphic_const_sum_groups:
                    self.writer.add_scalar("Train_Params/Homomorphic_K-hot", self.sum_groups, epoch)

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

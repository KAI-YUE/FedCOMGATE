import numpy as np
import logging
import copy

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Optimizer

# My libraries
from fedcom.buffer import WeightBuffer
from fedcom import quantizer_registry
from deeplearning import UserDataset

class LocalUpdater(object):
    def __init__(self, user_resource, config, **kwargs):
        """Construct a local updater for a user.

        Args:
            user_resources(dict):   a dictionary containing images and labels listed as follows. 
                - images (ndarry):  training images of the user.
                - labels (ndarray): training labels of the user.

            config (class):         global configuration containing info listed as follows:
                - lr (float):       learning rate for the user.
                - batch_size (int): batch size for the user. 
                - device (str):     set 'cuda' or 'cpu' for the user. 
                - predictor (str):  predictor type.
                - quantizer (str):  quantizer type.
        """
        
        try:
            self.lr = user_resource["lr"]
            self.momentum = user_resource["momentum"]
            self.weight_decay = user_resource["weight_decay"]
            self.batch_size = user_resource["batch_size"]
            self.device = user_resource["device"]
            
            assert("images" in user_resource)
            assert("labels" in user_resource)
        except KeyError:
            logging.error("LocalUpdater Initialization Failure! Input should include `lr`, `batch_size`!") 
        except AssertionError:
            logging.error("LocalUpdater Initialization Failure! Input should include samples!") 

        self.local_weight = None
        self.init_weight = None
        self.sample_loader = DataLoader(UserDataset(user_resource["images"], user_resource["labels"]), 
                                       batch_size=self.batch_size)

        self.criterion = nn.CrossEntropyLoss()
        self.quantizer = quantizer_registry[config.quantizer](config)
        self.tau = config.tau

    def local_step(self, model, offset):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model.
            offset(tensor):         delta offset term preventing client drift.
        """
        self.init_weight = copy.deepcopy(model.state_dict())
        # optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        tau_counter = 0
        break_flag = False

        offset_times_lr = offset * self.lr 

        while not break_flag:
            for sample in self.sample_loader:
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)
                optimizer.zero_grad()

                output = model(image)
                loss = self.criterion(output, label)
                loss.backward()
                optimizer.step()                        # w^(c+1) = w^(c) - \eta \hat{grad}

                self._offset(model, offset_times_lr)      # w^(c+1) = w^(c) - \eta \hat{grad} + \eta \delta

                tau_counter += 1
                if tau_counter > self.tau:
                    break_flag = True
                    break
        
        self.local_weight = copy.deepcopy(model.state_dict())
        
        # load back the model copy hence the global model won't be changed
        model.load_state_dict(self.init_weight)

    def _offset(self, model, offset_times_lr):
        model_buffer = WeightBuffer(model.state_dict())
        model_buffer = model_buffer - offset_times_lr
        model.load_state_dict(model_buffer.state_dict())

    def uplink_transmit(self):
        """Simulate the transmission of residual between local updated weight and local received initial weight.
        """ 
        try:
            assert(self.local_weight != None)
        except AssertionError:
            logging.error("No local model buffered!")

        # calculate the weight residual, then quantize and compress
        quantized_sets = [] 
        for w_name, w_pred in self.local_weight.items():
            residual = self.init_weight[w_name] - self.local_weight[w_name]
            quantized_set = self.quantizer.quantize(residual/self.lr)

            quantized_sets.append(quantized_set)

        local_package = quantized_sets
        return local_package


class GlobalUpdater(object):
    def __init__(self, config, initial_model, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:
                - quantizer (str):       quantizer type.

            initial_model (OrderedDict): initial model state_dict
        """
        self.quantizer = quantizer_registry[config.quantizer](config)

        self.num_users = int(config.users * config.sampling_fraction)
        self.lr = config.lr
        self.gamma = config.gamma
        self.debug_mode = config.debug_mode

        self.accumulated_delta = None
        self.local_residuals = None

    def global_step(self, model, local_packages, local_residual_buffers, **kwargs):
        """Perform a global update with collocted coded info from local users.
        """
        accumulated_delta = WeightBuffer(model.state_dict(), mode="zeros")
        accumulated_delta_state_dict = accumulated_delta.state_dict()

        global_model_state_dict = model.state_dict()
        for i, package in enumerate(local_packages):
            local_residuals_state_dict = local_residual_buffers[i].state_dict()
            for j, w_name in enumerate(global_model_state_dict):
                # dequantize
                quantized_sets = package[j]
                dequantized_residual = self.quantizer.dequantize(quantized_sets)
                local_residuals_state_dict[w_name] = dequantized_residual
                accumulated_delta_state_dict[w_name] += dequantized_residual
        
        accumulated_delta = accumulated_delta*(1/self.num_users)

        global_model = WeightBuffer(global_model_state_dict)
        global_model -= accumulated_delta*(self.lr*self.gamma)
        model.load_state_dict(global_model.state_dict())

        self.accumulated_delta = accumulated_delta

def update_offset_buffers(offset_buffers, local_residuals, accumulated_delta, tau):
    for i, offset in enumerate(offset_buffers):
        offset = offset + (local_residuals[i] - accumulated_delta)*(1/tau)
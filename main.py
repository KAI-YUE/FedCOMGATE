import os
import copy
import pickle
import logging
import numpy as np

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My libraries
from config import load_config
from config.utils import *
from deeplearning.dataset import *
from deeplearning.validate import *
from fedcom.myoptimizer import *
from fedcom.buffer import WeightBuffer

def train(model, config, logger, record):
    """Simulate Federated Learning training process. 
    
    Args:
        model (nn.Module):       the model to be trained.
        config (class):          the user defined configuration.
        logger (logging.logger): a logger for train info output.
        record (dict):           a record for train info saving.  
    """    
    # initialize userIDs
    users_to_sample = config.users
    userIDs = np.arange(config.users) 

    # initialize the optimizer for the server model
    dataset = assign_user_data(config, logger)

    # initialize the delta offset buffers and local residual buffers
    global_updater = GlobalUpdater(config, model.state_dict()) 

    # before optimization, report the result first
    validate_and_log(model, dataset, config, record, logger)
    
    for comm_round in range(config.rounds):
        userIDs_candidates = userIDs[:users_to_sample]
        
        # Wait for all users updating locally
        local_packages = []
        for i, user_id in enumerate(userIDs_candidates):
            user_resource = assign_user_resource(config, user_id, 
                                dataset["train_data"], dataset["user_with_data"])
            updater = LocalUpdater(user_resource, config)
            updater.local_step(model)
            local_package = updater.uplink_transmit()
            local_packages.append(local_package)

        # Update the global model
        global_updater.global_step(model, local_packages)

        # log and record
        logger.info("Round {:d}".format(comm_round))
        validate_and_log(model, dataset, config, record, logger)

        if comm_round == config.scheduler[0]:
            config.lr *= config.lr_scaler
            config.scheduler.pop(0)
        # config.lr *= 0.9

def main():
    config = load_config()
    
    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    logger = init_logger(config)
    model = init_model(config, logger)
    record = init_record(config, model)
    train(model, config, logger, record)
    save_record(config, record)

if __name__ == "__main__":
    main()


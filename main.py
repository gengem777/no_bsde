# This is a the script to implement the pricing algorithm

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import json
import munch
import time
import os

import numpy as np
import tensorflow as tf
import sde as eqn
import options as opts
import solvers as sls
from trainer import BSDETrainer, BermudanTrainer

sde_list = ["GBM", "TGBM", "SV", "CEV", "SVJ"]
option_list = ["European", "EuropeanPut", "Lookback", "Asian", "Basket", "BasketnoPI", "Swap", "TimeEuropean", "BermudanPut"]
dim_list = [1, 3, 5, 10, 20]

def main(sde_name: str, option_name: str, dim: int=1):
    if (sde_name not in sde_list) or (option_name not in option_list) or (dim not in dim_list):
        raise ValueError(f"please input right sde_name in {sde_list},\
                          option_name in {option_list} and dim in {dim_list}")
    else:
        json_path = f'./config/{sde_name}_{option_name}_{dim}.json'
    with open(json_path) as json_data_file:
        config = json.load(json_data_file)
    
    config = munch.munchify(config)
    initial_mode = config.eqn_config.initial_mode
    kernel_type = config.net_config.kernel_type
    sde = getattr(eqn, config.eqn_config.sde_name)(config)
    option = getattr(opts, config.eqn_config.option_name)(config)
    solver = getattr(sls, config.eqn_config.solver_name)(sde, option, config)
    if config.eqn_config.solver_name == "BermudanSolver":
        trainer = BermudanTrainer(solver)
    else:
        trainer = BSDETrainer(solver)
    
    print(f'begin train {option_name} under {sde_name} {dim} dimensions {kernel_type} {initial_mode}')
    checkpoint_path = f'./checkpoint/{sde_name}_{option_name}_{dim}_{initial_mode}_{kernel_type}'
    time_start = time.time()
    history = trainer.train(config.net_config.epochs, checkpoint_path)
    time_end = time.time()
    loss = history.losses
    print(f"time consume: {time_end - time_start}")
    np.savetxt(f'./train_curve/{option_name}_under_{sde_name}_{dim}train.txt', loss)
    print("----------end----------")
    

    
if __name__ == '__main__':
    main("GBM", "European", 1)
    # main("SV", "Swap", 10)
    # main("GBM", "Lookback", 1)
    # main("GBM", "Asian", 1)
    # main("GBM", "BasketnoPI", 10)
    # main("SV", "Swap", 1)
    # main("SV", "Swap", 10)
    # main("GBM", "BermudanPut", 3)


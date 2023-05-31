import json
import munch

import tensorflow as tf
import numpy as np
import sde as eqn
import options as opts
import solvers as sls
from trainer import BSDETrainer
from options import LookbackOption, GeometricAsian
from function_space import DeepONet, DeepONetwithPI

import matplotlib.pyplot as plt
from utils import mc_price
import os
print(os.getcwd())
sde_list = ["GBM", "TGBM", "SV", "CEV", "SVJ"]
option_list = ["European", "EuropeanPut", "Lookback", "Asian", "Basket", "Swap", "TimeEuropean", "TimeEuropeanBasketOption", "BermudanPut"]
dim_list = [1, 3, 5, 10, 20]

def get_useful_series(sde_name: str, option_name: str, dim: int=1, seed: int=0, training=True):
    tf.random.set_seed(seed)
    np.random.seed(seed)
   
    if (sde_name not in sde_list) or (option_name not in option_list) or (dim not in dim_list):
        raise ValueError(f"please input right sde_name in {sde_list},\
                          option_name in {option_list} and dim in {dim_list}")
    else:
        json_path = f'config/{sde_name}_{option_name}_{dim}.json'
    with open(json_path) as json_data_file:
        config = json.load(json_data_file)
    
    config = munch.munchify(config)
    sde = getattr(eqn, config.eqn_config.sde_name)(config)
    option = getattr(opts, config.eqn_config.option_name)(config)
    print(option)
    solver = getattr(sls, config.eqn_config.solver_name)(sde, option, config)
    trainer = BSDETrainer(solver)
    model = trainer.solver
    mode = config.eqn_config.initial_mode
    
    
    u_model = sde.sample_parameters(N=20, training=training)
    u_option = option.sample_parameters(N=20, training=training)
    u_hat = tf.concat([u_model, u_option], axis=-1)
    x, _ = sde.sde_simulation(u_hat, config.val_config.sample_size)
    time_stamp = tf.range(0, config.eqn_config.T, config.eqn_config.dt)
    time_steps = config.eqn_config.time_steps
    time_stamp = tf.reshape(time_stamp, [1, 1, time_steps, 1])
    t = tf.tile(time_stamp, [u_hat.shape[0], config.eqn_config.sample_size, 1, 1])
    u_hat = sde.expand_batch_inputs_dim(u_hat)
    # model.load_weights(f"checkpoint/{sde_name}_{option_name}_{dim}")
    print(config.net_config.kernel_type)


    model.no_net.load_weights(f"checkpoint/{sde_name}_{option_name}_{dim}_{mode}_{config.net_config.kernel_type}_pi")
    if type(option) == LookbackOption or type(option) == GeometricAsian:
        x_m = option.markovian_var(x)
        x_arg = tf.concat([x, x_m], axis=-1)
    else: 
        x_arg = x
    print(f"asset's shepe: {x_arg.shape}")
    y_pred = model.net_forward((t, x_arg, u_hat))
    # z_pred = tf.squeeze(model.z_hedge(t, x_arg, u_hat)).numpy()
    y_pred = tf.squeeze(y_pred).numpy()
    t_test = tf.squeeze(t).numpy()
    x_mc = tf.squeeze(np.mean(x_arg[:,:,:,:dim], axis=-1)).numpy()
    # x_mc = np.mean(x_mc, axis=-1)
    u = tf.squeeze(u_hat[:, 0,0, :]).numpy()
    print(f"parameters shape: {u.shape}")

    mc_p = mc_price(x_arg[:,:,:,:dim], u_hat[:, 0,0, :])

    if (sde_name in ["GBM", "TGBM", "SV"]) and (option_name in ["European", "EuropeanPut","Swap", "TimeEuropean", "BermudanPut"]) and (dim in[1, 3, 5, 10]):
        y_true = option.exact_price(t, x, u_hat)
        # z_true = tf.squeeze(option.exact_delta(t, x, u_hat)).numpy()
        y_true = tf.squeeze(y_true).numpy()
    
        return y_pred, x_mc, t_test, u, y_true, _, _
    
    if (sde_name in [ "SV"]) and (option_name in [ "Swap"]) and (dim != 1):
        y_true = option.exact_price(t, x, u_hat)
        # z_true = tf.squeeze(option.exact_delta(t, x, u_hat)).numpy()
        y_true = tf.squeeze(y_true).numpy()
    
        return y_pred, x_mc, t_test, u, y_true, _, _
    
    elif (sde_name == "GBM") and (option_name in ["Lookback", "Asian"]) and (dim in [1, 3, 10]):
        y_true = option.exact_price(t, x_arg, u_hat)
        # z_true = tf.squeeze(option.exact_delta(t, x_arg, u_hat)).numpy()
        y_true = tf.squeeze(y_true).numpy()
        return y_pred, x_mc, t_test, u, y_true, _, _

    
    elif sde_name == "GBM" and option_name == "Basket_wo_PI" and dim != 1:
        y_true = option.exact_price(t, x_arg, u_hat)
        y_true = tf.squeeze(y_true).numpy()
        x_mc = tf.reduce_mean(x_arg, axis=-1)
        x_mc = tf.squeeze(x_mc).numpy()
        return y_pred, x_mc, t_test, u, y_true
    
    elif sde_name == "GBM" and option_name == "Basket" and dim != 1:
        y_true = option.exact_price(t, x_arg, u_hat)
        y_true = tf.squeeze(y_true).numpy()
        x_mc = tf.reduce_mean(x_arg, axis=-1)
        x_mc = tf.squeeze(x_mc).numpy()
        return y_pred, x_mc, t_test, u, y_true, _, _
    
    else:
        return y_pred, x_mc, t_test, u, mc_p.numpy()



data = get_useful_series("GBM", "Basket", 10, 0, False)     
# print(data[1][0, :, 0]) 
print(data[0].shape, data[4].shape)
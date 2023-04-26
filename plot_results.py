import tensorflow as tf
import numpy as np
from config import Config
from sde import GeometricBrowianMotion, CEVModel, HestonModel
from options import EuropeanOption, LookbackOption, GeometricAsian
from function_space import DeepONet, DeepONetwithPI
import matplotlib.pyplot as plt
import os

def get_useful_series(sde_name: str, option_name: str, dim: int=1, seed: int=0):
    config = Config(dim=dim, x_init=1, iid=False)
    print(config.x_init)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    if sde_name == 'GBM':
        sde = GeometricBrowianMotion(config)
    elif sde_name == "CEV":
        sde = CEVModel(config)
    elif sde_name == "SV":
        sde = HestonModel(config)
    else:
        raise ValueError("wrong SDE name")
    if option_name == 'European':
        option = EuropeanOption(config)
    elif option_name == 'Asian':
        option = GeometricAsian(config)
    elif option_name == 'Lookback':
        option = LookbackOption(config)
    elif option_name == 'Basket':
        option = EuropeanBasketOption(config)
    else:
        raise ValueError("wrong OPTION name")
    if config.iid == True:
        model = DeepONet([config.n_hidden] *
                    config.n_layers, [config.n_hidden] * config.n_layers)
    else:
        pi_layers = [15, 15, 15]
        model = DeepONetwithPI([config.n_hidden] *
                    config.n_layers, [config.n_hidden] * config.n_layers, pi_layers, dim)
    print(option.m_range)
    print(type(option))
        
    u_model = sde.sample_parameters(N=1)
    u_option = option.sample_parameters(N=1)
    u_hat = tf.concat([u_model, u_option], axis=-1)
    x, _ = sde.sde_simulation(u_hat, config.M)
    time_stamp = tf.range(0, config.T, config.dt)
    time_steps = int(config.T / config.dt)
    time_stamp = tf.reshape(time_stamp, [1, 1, time_steps, 1])
    t = tf.tile(time_stamp, [u_hat.shape[0], config.M, 1, 1])
    u_hat = sde.expand_batch_inputs_dim(u_hat)
    model.load_weights(f"checkpoint/{sde_name}_{option_name}_{dim}")
    if type(option) == LookbackOption or type(option) == GeometricAsian:
        x_m = option.markovian_var(x)
        x_arg = tf.concat([x, x_m], axis=-1)
    else: 
        x_arg = x
    print(model)

    y_pred = model((t, x_arg, u_hat))
    y_pred = tf.squeeze(y_pred).numpy()
    t_test = tf.squeeze(t).numpy()
    x_mc = tf.squeeze(x_arg[:,:,:,:dim]).numpy()
    u = tf.squeeze(u_hat[:, 0,0, :]).numpy()

    if sde_name == "GBM" and option_name == "European" and dim == 1:
        y_true = option.exact_price(t, x, u_hat)
        y_true = tf.squeeze(y_true).numpy()
    
        return y_pred, x_mc, t_test, u, y_true
    
    elif sde_name == "GBM" and option_name == "Lookback" and dim == 1:
        y_true = option.exact_price(t, x_arg, u_hat)
        print(y_true)
        y_true = tf.squeeze(y_true).numpy()
        return y_pred, x_mc, t_test, u, y_true
    
    elif sde_name == "GBM" and option_name == "Asian" and dim == 1:
        y_true = option.exact_price(t, x_arg, u_hat)
        y_true = tf.squeeze(y_true).numpy()
        return y_pred, x_mc, t_test, u, y_true
    
    
    elif sde_name == "GBM" and option_name == "Basket" and dim != 1:
        y_true = option.exact_price((t, x_arg, u_hat))
        y_true = tf.squeeze(y_true).numpy()
        return y_pred, x_mc, t_test, u, y_true
    
    else:
        return y_pred, x_mc, t_test, u



data = get_useful_series("GBM", "European", 10, 0)     
# print(data[1][0, :, 0]) 
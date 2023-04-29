import tensorflow as tf
import numpy as np
from config import Config
from sde import GeometricBrowianMotion, CEVModel, HestonModel
from options import EuropeanOption, LookbackOption, GeometricAsian
from function_space import DeepONet, DeepONetwithPI
from markov_solver import MarkovianSolver
from utils import mc_price
import matplotlib.pyplot as plt
import os

def get_useful_series(sde_name: str, option_name: str, dim: int=1, seed: int=0):
    config = Config(dim=dim, x_init=1, iid=True)
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
    elif option_name == 'Basket' or 'Basket_wo_PI' or "Basket_PI":
        option = EuropeanBasketOption(config)
    else:
        raise ValueError("wrong OPTION name")
    # if config.iid == True:
    #     if config.dim != 1:
    #         model = DeepONet([50] * 4, [50] * 4)
    #     else:
    #         model = DeepONet([15] * 4, [15] * 4)
    # else:
    #     # pi_layers = [15, 15, 15]
    #     pi_layers = [10, 10]
    #     # model = DeepONetwithPI([15] * 4, [15] * 4, pi_layers, dim)
    #     model = DeepONetwithPI([20] * 4, [20] * 4, pi_layers, dim)
    # print(option.m_range)
    # print(type(option))
    m_model = MarkovianSolver(sde, option, config)
        
    u_model = sde.sample_parameters(N=1)
    u_option = option.sample_parameters(N=1)
    u_hat = tf.concat([u_model, u_option], axis=-1)
    x, _ = sde.sde_simulation(u_hat, config.M)
    time_stamp = tf.range(0, config.T, config.dt)
    time_steps = int(config.T / config.dt)
    time_stamp = tf.reshape(time_stamp, [1, 1, time_steps, 1])
    t = tf.tile(time_stamp, [u_hat.shape[0], config.M, 1, 1])
    u_hat = sde.expand_batch_inputs_dim(u_hat)
    # model.load_weights(f"checkpoint/{sde_name}_{option_name}_{dim}")
    m_model.no_net.load_weights(f"checkpoint/{sde_name}_{option_name}_{dim}")
    if type(option) == LookbackOption or type(option) == GeometricAsian:
        x_m = option.markovian_var(x)
        x_arg = tf.concat([x, x_m], axis=-1)
    else: 
        x_arg = x
    print(m_model.no_net)
    print(x_arg.shape)
    y_pred = m_model.net_forward((t, x_arg, u_hat))
    y_pred = tf.squeeze(y_pred).numpy()
    t_test = tf.squeeze(t).numpy()
    x_mc = tf.squeeze(x_arg[:,:,:,:dim]).numpy()
    u = tf.squeeze(u_hat[:, 0,0, :]).numpy()

    mc_p = mc_price(x_arg[:,:,:,:dim], u_hat[:, 0,0, :])
    print(mc_p)

    if sde_name == "GBM" and option_name == "European" and dim == 1:
        y_true = option.exact_price(t, x, u_hat)
        y_true = tf.squeeze(y_true).numpy()
    
        return y_pred, x_mc, t_test, u, mc_p.numpy()
    
    elif sde_name == "GBM" and option_name == "Lookback" and dim == 1:
        y_true = option.exact_price(t, x_arg, u_hat)
        print(y_true)
        y_true = tf.squeeze(y_true).numpy()
        return y_pred, x_mc, t_test, u, y_true
    
    elif sde_name == "GBM" and option_name == "Asian" and dim == 1:
        y_true = option.exact_price(t, x_arg, u_hat)
        y_true = tf.squeeze(y_true).numpy()
        return y_pred, x_mc, t_test, u, y_true
    
    
    elif sde_name == "GBM" and option_name == "Basket_wo_PI" and dim != 1:
        y_true = option.exact_price(t, x_arg, u_hat)
        y_true = tf.squeeze(y_true).numpy()
        print(x_mc.shape)
        x_mc = tf.reduce_mean(x_arg, axis=-1)
        x_mc = tf.squeeze(x_mc).numpy()
        return y_pred, x_mc, t_test, u, y_true
    
    elif sde_name == "GBM" and option_name == "Basket_PI" and dim != 1:
        y_true = option.exact_price(t, x_arg, u_hat)
        y_true = tf.squeeze(y_true).numpy()
        print(x_mc.shape)
        x_mc = tf.reduce_mean(x_arg, axis=-1)
        x_mc = tf.squeeze(x_mc).numpy()
        return y_pred, x_mc, t_test, u, y_true
    
    else:
        return y_pred, x_mc, t_test, u, mc_p.numpy()



data = get_useful_series("GBM", "European", 1, 0) 
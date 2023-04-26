import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from function_space import DeepONet, DeepONetwithPI, DeepKernelONetwithPI
from typing import List, Tuple
import time


class BaseBSDESolver(tf.keras.Model):
    """
    This is the class to construct the tain step of the BSDE loss function, the data is generated from three
    external classes: the sde class which yield the data of input parameters. the option class which yields the
    input function data and give the payoff function and exact option price.
    """

    def __init__(self, sde, option, config):
        super(BaseBSDESolver, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.option = option
        self.sde = sde
        self.dim = self.eqn_config.dim
        self.branch_layers = self.net_config.branch_layers
        self.trunk_layers = self.net_config.trunk_layers
        self.filters = self.net_config.filters
        self.strides = self.net_config.strides
        self.pi_layers = self.net_config.pi_layers
        if self.net_config.kernel_type == "dense":
            self.no_net = DeepKernelONetwithPI(branch_layer=self.branch_layers, 
                                                trunk_layer=self.trunk_layers, 
                                                pi_layer=self.pi_layers, 
                                                num_assets=self.dim, 
                                                dense=True, 
                                                num_outputs=6)
        else:                                 
            self.no_net = DeepKernelONetwithPI(branch_layer=self.branch_layers, 
                                                trunk_layer=self.trunk_layers, 
                                                pi_layer=self.pi_layers, 
                                                num_assets=self.dim, 
                                                dense=False, 
                                                num_outputs=6,
                                                filters=self.filters, 
                                                strides=self.strides)

        self.time_horizon = self.eqn_config.T
        self.batch_size = self.eqn_config.batch_size
        self.samples = self.eqn_config.sample_size
        self.dt = self.eqn_config.dt
        self.time_steps = self.eqn_config.time_steps
        time_stamp = tf.range(0, self.time_horizon, self.dt)
        time_stamp = tf.reshape(time_stamp, [1, 1, self.time_steps, 1])
        self.time_stamp = tf.tile(time_stamp, [self.batch_size, self.samples, 1, 1])
        self.alpha = self.net_config.alpha

    def net_forward(self, inputs: Tuple[tf.Tensor]) -> tf.Tensor:
        t, x, u = inputs
        u_c, u_p = self.sde.split_uhat(u)
        y = self.no_net((t, x, u_c, u_p))
        return y
        

    def call(self, data: Tuple[tf.Tensor], training=None):
        raise NotImplementedError

    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor]) -> dict:
        raise NotImplementedError

    def h_tf(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, param: tf.Tensor) -> tf.Tensor:  # get h function
        raise NotImplementedError

    def z_tf(self, t: tf.Tensor, x: tf.Tensor, grad: tf.Tensor, dw: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def g_tf(self, x: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

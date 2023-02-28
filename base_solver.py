import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from function_space import DeepONet
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
        self.config = config
        self.option = option
        self.sde = sde
        self.dim = self.config.dim
        self.n_hidden = self.config.n_hidden
        self.n_layers = self.config.n_layers
        self.no_net = DeepONet([self.n_hidden] *
                               self.n_layers, [self.n_hidden] * self.n_layers)
        self.time_horizon = self.config.T
        self.batch_size = self.config.batch_size
        self.samples = self.config.M
        self.dt = self.config.dt
        self.time_steps = int(self.time_horizon / self.dt)
        time_stamp = tf.range(0, self.time_horizon, self.dt)
        time_stamp = tf.reshape(time_stamp, [1, 1, self.time_steps, 1])
        self.time_stamp = tf.tile(time_stamp, [self.batch_size, self.samples, 1, 1])
        self.loss_curve = []
        self.alpha = self.config.alpha

    def net_forward(self, t: tf.Tensor, x: tf.Tensor, par:
    tf.Tensor) -> tf.Tensor:
        pass

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

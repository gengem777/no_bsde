import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from function_space import DeepONet
from base_solver import BaseBSDESolver
from data_generators import DiffusionModelGenerator
from options import EuropeanOption
from typing import List, Tuple
from sde import GeometricBrownianMotion, CEVModel, HestonModel
import time
import os


class MarkovianSolver(BaseBSDESolver):
    """
    This is the class to construct the tain step of the BSDE loss function, the data is generated from three
    external classes: the sde class which yield the data of input parameters. the option class which yields the
    input function data and give the payoff function and exact option price.
    """

    def __init__(self, sde, option, config):
        super(MarkovianSolver, self).__init__(sde, option, config)

    def call(self, data: Tuple[tf.Tensor], training=None) -> Tuple[tf.Tensor]:
        """
        :param t: time B x M x (T-1) x 1
        :param x: state B x M x (T-1) x D
        :param  param: B x K ->(repeat) B x M x (T-1) x K
        :return: value (B, M, (T-1), 1), gradient_x (B, M, (T-1), D)
        """
        t, x, dw, u_hat = data
        N = self.eqn_config.time_steps
        loss_interior = 0.0
        u_now = u_hat[:, :, :-1, :]
        u_pls = u_hat[:, :, 1:, :]
        x_now = x[:, :, :-1, :]
        x_pls = x[:, :, 1:, :]
        t_now = t[:, :, :-1, :]
        t_pls = t[:, :, 1:, :]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_now)
            f_now = self.net_forward((t_now, x_now, u_now))
            grad = tape.gradient(f_now, x_now) # grad wrt whole Markovian variable (X_t, M_t) 
                                            # First self.dim entry is the grad w.r.t X_t
        f_pls = self.net_forward((t_pls, x_pls, u_pls))
        for n in range(N-1):
            V_pls = f_pls[:, :, n:, :]
            V_now = f_now[:, :, n:, :]
            V_hat = V_now - self.h_tf(t_now[:,:, n:,:], x_now[:,:, n:,:], V_now, u_now[:,:, n:,:]) * self.dt + self.z_tf(t_now[:,:, n:,:], x_now[:,:, n:,:], 
                                                                            grad[:,:, n:,:], dw[:,:, n:,:], u_now[:,:, n:,:])
            tele_sum = tf.reduce_sum(V_pls - V_hat, axis=2)
            loss_interior += tf.reduce_mean(tf.square(tele_sum + self.g_tf(x, u_hat) - f_pls[:, :, -1, :]))
        loss_tml = tf.reduce_mean(tf.square(f_pls[:, :, -1, :] - self.g_tf(x, u_hat)))
        loss = self.alpha * loss_tml + loss_interior
        return loss, loss_interior, loss_tml

    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor]) -> dict:
        with tf.GradientTape() as tape:
            with tf.name_scope('calling_model'):
                loss, loss_int, loss_tml = self(inputs[0])
            loss, loss_int, loss_tml = tf.reduce_mean(loss), tf.reduce_mean(loss_int), tf.reduce_mean(loss_tml)
            grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {"loss": loss,
                "loss interior": loss_int,
                "loss terminal": loss_tml}
    
    def loss(self, inputs):
        loss, _, _ = self(inputs[0])
        loss = tf.reduce_mean(loss)
        return loss

    def h_tf(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, param: tf.Tensor) -> tf.Tensor:  # get h function
        """
        the driver term is r*y for MTG property
        """
        r = tf.expand_dims(param[:, :, :, 0], -1)
        return r * y

    def z_tf(self, t: tf.Tensor, x: tf.Tensor, grad: tf.Tensor, dw: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        grad: (B, M, N-1, d)
        dw: (B, M, N-1, d)
        give: \sigma(t, x) * grad
        for a batch of (t, x, par)
        """
        # v_tx = self.sde.diffusion_onestep(t, x, u_hat)
        if not isinstance(self.sde, HestonModel):
            x = x[...,:self.dim]
            grad = grad[...,:self.dim]
            v_tx = self.sde.diffusion_onestep(t, x[...,:self.dim], u_hat)
        else:
            v_tx = self.sde.diffusion_onestep(t, x, u_hat)
        z = tf.reduce_sum(v_tx * grad * dw, axis=-1, keepdims=True)
        return z

    def g_tf(self, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        payoff = self.option.payoff(x, u_hat)
        return payoff


class BSDEMarkovianModel:
    def __init__(self, sde, option, config):
        self.sde = sde
        self.config = config # whole config
        self.option = option

    def pre_setting(self):
        learning_rate = self.config.net_config.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=200,
            decay_rate=0.9
        )
        Optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
        self.model = MarkovianSolver(self.sde, self.option, self.config)
        self.data_generator = DiffusionModelGenerator(self.sde, self.config, self.option, 100)
        # self.val_generator = DiffusionModelGenerator(self.sde, self.config, self.option, 20)
        self.model.compile(optimizer=Optimizer)

    def train(self, nr_epochs: int, checkpoint_path: str):
        self.pre_setting()
        # Create a callback that saves the model's batch loss
        class LossHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []
            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
        history = LossHistory()
        self.model.fit(x=self.data_generator, epochs=nr_epochs, callbacks=[history])
        self.model.no_net.save_weights(checkpoint_path)
        # return history
        return history
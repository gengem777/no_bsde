import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from function_space import DeepONet
from base_solver import BaseBSDESolver
from data_generators import DiffusionModelGenerator
from typing import List, Tuple
from sde import GeometricBrowianMotion, CEVModel, HestonModel
import time


class MarkovianSolver(BaseBSDESolver):
    """
    This is the class to construct the tain step of the BSDE loss function, the data is generated from three
    external classes: the sde class which yield the data of input parameters. the option class which yields the
    input function data and give the payoff function and exact option price.
    """

    def __init__(self, sde, option, config):
        super(MarkovianSolver, self).__init__(sde, option, config)

    def net_forward(self, t: tf.Tensor, x: tf.Tensor, par: tf.Tensor) -> tf.Tensor:
        inputs = t, x, par
        y = self.no_net(inputs)
        return y

    def call(self, data: Tuple[tf.Tensor], training=None) -> Tuple[tf.Tensor]:
        """
        :param t: time B x M x (T-1) x 1
        :param x: state B x M x (T-1) x D
        :param  param: B x K ->(repeat) B x M x (T-1) x K
        :return: value (B, M, (T-1), 1), gradient_x (B, M, (T-1), D)
        """
        t_pls, x_pls, t_now, x_now, x, dw, u_hat = data
        u_now = u_hat[:, :, :-1, :]
        u_pls = u_hat[:, :, 1:, :]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_now)
            f_now = self.net_forward(t_now, x_now, u_now)
            grad = tape.gradient(f_now, x_now) # grad wrt whole Markovian variable (X_t, M_t) 
                                               # First self.dim entry is the grad w.r.t X_t
        
        f_pls = self.net_forward(t_pls, x_pls, u_pls)
        f_hat = f_now - self.h_tf(t_now, x_now, f_now, u_now) * self.dt + self.z_tf(t_now, x_now, 
                                                                        grad, dw, u_now)
        loss_int = self.alpha * tf.reduce_mean(tf.square(f_hat - f_pls), 2)

        # region
        # else:
        #     f_pls = self.net_forward(t_pls, x_pls, par_pls)
        #     x_pls_1 = self.sde.euler_onestep(t_now, x_now, dw1, par_now)
        #     f_pls_1 = self.net_forward(t_pls, x_pls_1, par_pls)
        #     x_pls_2 = self.sde.euler_onestep(t_now, x_now, dw2, par_now)
        #     f_pls_2 = self.net_forward(t_pls, x_pls_2, par_pls)
        #     f_hat_1 = f_now - self.h_tf(t_now, x_now, f_now, par_now) * self.dt + self.z_tf(t_now, x_now, grad, dw1,
        #                                                                                     par_now)
        #     f_hat_2 = f_now - self.h_tf(t_now, x_now, f_now, par_now) * self.dt + self.z_tf(t_now, x_now, grad, dw2,
        #                                                                                     par_now)
        #     loss_int = self.alpha * tf.reduce_mean((f_hat_1 - f_pls_1) * (f_hat_2 - f_pls_2))
        # endregion

        if self.config.initial_mode != 'random':
            loss_int += tf.reduce_mean(tf.math.reduce_variance(f_now[:, :, 0, :], 1))
        loss_tml = tf.square(f_pls[:, :, -1, :] - self.g_tf(x, u_hat))
        loss = loss_int + loss_tml
        return f_now, loss, loss_int, loss_tml

    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor]) -> dict:
        with tf.GradientTape() as tape:
            with tf.name_scope('calling_model'):
                _, loss, loss_int, loss_tml = self(inputs[0])
                # tf.print(loss, loss_int, loss_tml)
            loss, loss_int, loss_tml = tf.reduce_mean(loss), tf.reduce_mean(loss_int), tf.reduce_mean(loss_tml)
        
            grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        # y_true = inputs[1]
        # error = tf.reduce_mean(tf.math.abs(y_true - y))
        return {"loss": loss,
                "loss interior": loss_int,
                "loss terminal": loss_tml}
                # "price error": error}
    
    def grad(self, inputs):
        with tf.GradientTape() as tape:
            _, loss, _, _ = self(inputs[0])
                # tf.print(loss, loss_int, loss_tml)
            loss = tf.reduce_mean(loss)
            grad = tape.gradient(loss, self.trainable_variables)
        return loss, grad

    def h_tf(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, param: tf.Tensor) -> tf.Tensor:  # get h function
        """
        the driver term is r*y for MTG property
        """
        r = tf.expand_dims(param[:, :, :, 0], -1)
        return r * y

    def z_tf(self, t: tf.Tensor, x: tf.Tensor, grad: tf.Tensor, dw: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        """
        grad: (B, M, N-1, d)
        dw: (B, M, N-1, d)
        give: \sigma(t, x) * grad
        for a batch of (t, x, par)
        """
        v_tx = self.sde.diffusion_onestep(t, x, param)
        if not isinstance(self.sde, HestonModel):
            x = x[...,:self.dim]
            grad = grad[...,:self.dim]
        z = tf.reduce_sum(v_tx * grad * dw, axis=-1, keepdims=True)
        return z

    def g_tf(self, x: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        payoff = self.option.payoff(x, param)
        return payoff


class BSDEMarkovianModel:
    def __init__(self, sde, option, config):
        self.sde = sde
        self.config = config
        self.option = option

    def pre_setting(self):
        learning_rate = self.config.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=200,
            decay_rate=0.9
        )
        Optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
        self.model = MarkovianSolver(self.sde, self.option, self.config)
        self.data_generator = DiffusionModelGenerator(self.sde, self.config, self.option)
        self.model.compile(optimizer=Optimizer)

    def train(self, nr_epochs: int):
        self.pre_setting()
        self.model.fit(x=self.data_generator, epochs=nr_epochs)

    def grad_norm(self):
        self.model

    def save_asset_price_series(self):
        seed = 600
        tf.random.set_seed(seed)
        np.random.seed(seed)
        r = np.random.uniform(0.038, 0.052)
        s = np.random.uniform(0.2, 0.6)
        m = np.random.normal(1.0, 0.1)
        params_eval = tf.constant([[r, s, m]])
        x, _, _, _ = self.sde.sde_simulation(params_eval, self.config.M)
        time_stamp = tf.range(0, self.config.T, self.config.dt)
        time_steps = int(self.config.T / self.config.dt)
        time_stamp = tf.reshape(time_stamp, [1, 1, time_steps, 1])
        t = tf.tile(time_stamp, [1, self.config.M, 1, 1])
        params_eval = tf.reshape(params_eval, [1, 1, 1, 3])
        params_eval = tf.tile(params_eval, [1, self.config.M, self.config.time_steps, 1])
        y_true = self.option.exact_price(t, x, params_eval)
        y_pred = self.model.net_forward(t, x, params_eval)
        y_true = tf.squeeze(y_true).numpy()
        y_pred = tf.squeeze(y_pred).numpy()
        t_test = tf.squeeze(t).numpy()
        np.save('y_true.npy', y_true)
        np.save('y_pred.npy', y_pred)
        np.save('t.npy', t_test)
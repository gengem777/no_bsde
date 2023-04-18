import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from function_space import DeepONet
from base_solver import BaseBSDESolver
from data_generators import DiffusionModelGenerator
from options import EuropeanOption
from typing import List, Tuple
from sde import GeometricBrowianMotion, CEVModel, HestonModel
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
        t, x, dw, u_hat = data
        N = self.config.time_steps
        loss_interior = 0.0
        u_now = u_hat[:, :, :-1, :]
        u_pls = u_hat[:, :, 1:, :]
        x_now = x[:, :, :-1, :]
        x_pls = x[:, :, 1:, :]
        t_now = t[:, :, :-1, :]
        t_pls = t[:, :, 1:, :]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_now)
            f_now = self.no_net((t_now, x_now, u_now))
            grad = tape.gradient(f_now, x_now) # grad wrt whole Markovian variable (X_t, M_t) 
                                            # First self.dim entry is the grad w.r.t X_t
        f_pls = self.no_net((t_pls, x_pls, u_pls))
        for n in range(N-1):
            V_pls = f_pls[:, :, n:, :]
            V_now = f_now[:, :, n:, :]
            V_hat = V_now - self.h_tf(t_now[:,:, n:,:], x_now[:,:, n:,:], V_now, u_now[:,:, n:,:]) * self.dt + self.z_tf(t_now[:,:, n:,:], x_now[:,:, n:,:], 
                                                                            grad[:,:, n:,:], dw[:,:, n:,:], u_now[:,:, n:,:])
            tele_sum = tf.reduce_sum(V_pls - V_hat, axis=2)
            loss_interior += tf.reduce_mean(tf.square(tele_sum + self.g_tf(x, u_hat) - f_pls[:, :, -1, :]))
        loss_tml = tf.reduce_mean(tf.square(f_pls[:, :, -1, :] - self.g_tf(x, u_hat)))
        loss = self.alpha * loss_tml + loss_interior




        # u_now = u_hat[:, :, :-1, :]
        # u_pls = u_hat[:, :, 1:, :]
        # with tf.GradientTape(watch_accessed_variables=False) as tape:
        #     tape.watch(x_now)
        #     f_now = self.no_net((t_now, x_now, u_now))
        #     grad = tape.gradient(f_now, x_now) # grad wrt whole Markovian variable (X_t, M_t) 
        #                                        # First self.dim entry is the grad w.r.t X_t
        
        # f_pls = self.no_net((t_pls, x_pls, u_pls))
        # f_hat = f_now - self.h_tf(t_now, x_now, f_now, u_now) * self.dt + self.z_tf(t_now, x_now, 
        #                                                                 grad, dw, u_now)
        # loss_int = self.alpha * tf.reduce_mean(tf.square(f_hat - f_pls), 2)
        # tele_sum = tf.reduce_sum(f_pls - f_hat, axis=2)
        # # loss_int = tf.reduce_mean(tele_sum)
        # # region
        # # else:
        # #     f_pls = self.net_forward(t_pls, x_pls, par_pls)
        # #     x_pls_1 = self.sde.euler_onestep(t_now, x_now, dw1, par_now)
        # #     f_pls_1 = self.net_forward(t_pls, x_pls_1, par_pls)
        # #     x_pls_2 = self.sde.euler_onestep(t_now, x_now, dw2, par_now)
        # #     f_pls_2 = self.net_forward(t_pls, x_pls_2, par_pls)
        # #     f_hat_1 = f_now - self.h_tf(t_now, x_now, f_now, par_now) * self.dt + self.z_tf(t_now, x_now, grad, dw1,
        # #                                                                                     par_now)
        # #     f_hat_2 = f_now - self.h_tf(t_now, x_now, f_now, par_now) * self.dt + self.z_tf(t_now, x_now, grad, dw2,
        # #                                                                                     par_now)
        # #     loss_int = self.alpha * tf.reduce_mean((f_hat_1 - f_pls_1) * (f_hat_2 - f_pls_2))
        # # endregion

        # # if self.config.initial_mode != 'random':
        # #     loss_int += tf.reduce_mean(tf.math.reduce_variance(f_now[:, :, 0, :], 1))
        # loss_tml = tf.reduce_mean(tf.square(f_pls[:, :, -1, :] - self.g_tf(x, u_hat)))
        # loss_1 = loss_tml
        # loss_2 = tf.reduce_mean(tf.square(tele_sum + self.g_tf(x, u_hat) - f_pls[:, :, -1, :]))
        # loss = self.alpha * loss_1 + loss_2
        return loss, loss_interior, loss_tml

    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor]) -> dict:
        with tf.GradientTape() as tape:
            with tf.name_scope('calling_model'):
                loss, loss_int, loss_tml = self(inputs[0])
                # tf.print(loss, loss_int, loss_tml)
            loss, loss_int, loss_tml = tf.reduce_mean(loss), tf.reduce_mean(loss_int), tf.reduce_mean(loss_tml)
        
            grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        # y_true = inputs[1]
        # error = tf.reduce_mean(tf.math.abs(y_true - y))
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

    def save_asset_price_series(self, sde_name: str, option_name: str):
        # seed = 600
        # tf.random.set_seed(seed)
        # np.random.seed(seed)
        # dim = self.config.dim
        # u_model = self.sde.sample_parameters(N=1)
        # u_option = self.option.sample_parameters(N=1)
        # u_hat = tf.concat([u_model, u_option], axis=-1)
        # x, _ = self.sde.sde_simulation(u_hat, self.config.M)
        # time_stamp = tf.range(0, self.config.T, self.config.dt)
        # time_steps = int(self.config.T / self.config.dt)
        # time_stamp = tf.reshape(time_stamp, [1, 1, time_steps, 1])
        # t = tf.tile(time_stamp, [u_hat.shape[0], self.config.M, 1, 1])
        # u_hat = self.sde.expand_batch_inputs_dim(u_hat)
        # y_pred = self.model.net_forward(t, x, u_hat)
        # y_pred = tf.squeeze(y_pred).numpy()
        # t_test = tf.squeeze(t).numpy()
        # x_mc = tf.squeeze(x).numpy
        # print(y_pred[:, 1, 0])
        # y_mc = tf.nn.relu(tf.reduce_mean(x[:,:,-1,:], -1) - u_hat[:, :, -1, -1]) #(B, M)
        # y_mc = tf.exp(-u_hat[:,:, -1, 0]) * y_mc
        # y_mc = tf.reduce_mean(y_mc, axis=1)
        # print(y_mc.numpy())

        # np.save(f'predicted_price/{sde_name}_{option_name}_{dim}_yhat.npy', y_pred)
        # np.save('predicted_price/t.npy', t_test)
        # np.save(f'predicted_price/u_hat_{sde_name}_{option_name}_{dim}.npy', u_hat[:,0,0,:])
        # np.save(f'predicted_price/x_{sde_name}_{option_name}_{dim}.npy', x_mc)

        # if sde_name == "GBM" and option_name == "European" and dim == 1:
        #     y_true = self.option.exact_price(t, x, u_hat)
        #     y_true = tf.squeeze(y_true).numpy()
        #     np.save(f'predicted_price/{sde_name}_{option_name}_{dim}_ytrue.npy', y_true)
        #     # print(y_true, y_pred)
        pass
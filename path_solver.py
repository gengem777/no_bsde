import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from function_space import DeepONet
from base_bsde_solver import BaseBSDESolver
from data_generators import DiffusionModelGenerator
from typing import List, Tuple
import time

class MarkovianSolver(BaseBSDESolver):
    """
    This is the class to construct the tain step of the BSDE loss function, the data is generated from three
    external classes: the sde class which yield the
    """
    def __init__(self, sde, option, config):
        super(MarkovianSolver, self).__init__(sde, option, config)
        self.n_a = self.config.n_a
        self.RNNCells = tf.keras.layers.LSTM(self.n_a, activation='tanh', return_sequences=True, return_state=False)

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
        t_pls, x_pls, t_now, x_now, x, dw, param, dw1, dw2 = data
        par_now = param[:, :, :-1, :]
        par_pls = param[:, :, 1:, :]
        batch = tf.shape(x)[0]
        sample = tf.shape(x)[1]
        steps = tf.shape(x)[2] - 1
        batch_num = batch * sample
        sequence_x_t = tf.reshape(tf.concat([t_now, x_now], axis=-1), [batch_num, steps, 1+self.dim])
        a = tf.zeros([batch_num, self.n_a])  # (B*M, hidden)
        c = tf.zeros([batch_num, self.n_a])
        a_hidden = self.RNNCells(sequence_x_t, initial_state=[a, c]) 
        a = tf.expand_dims(a, 1)  # (batch_num, 1, hidden)
        a_hidden_1 = tf.concat([a, a_hidden[:, :, :]], axis=1)  # (B*M, T+1, hid_dim)
        a_pls = tf.reshape(a_hidden_1[:, 1:, :], [batch, sample, steps, self.n_a]) 
        a_now = tf.reshape(a_hidden_1[:, :-1, :], [batch, sample, steps, self.n_a]) 
        with tf.GradientTape() as tape:
            tape.watch(x_now)
            xa_now = tf.concat([x_now, a_now], axis=-1)
            f_now = self.NOforward(t_now, xa_now, par_now)
            grad = tape.gradient(f_now, x_now)
    
        if self.config.is_Maliar == False:
                xa_pls = tf.concat([x_pls, a_pls], axis=-1)
                f_pls = self.NOforward(t_pls, xa_pls, par_pls)
                f_hat = f_now - self.h_tf(t_now, x_now, f_now, par_now) * self.dt + self.z_tf(t_now, x_now, grad, dw, par_now)
                loss_int = self.alpha * tf.reduce_mean(tf.square(f_hat - f_pls), 2)
            
        else:
            xa_pls = tf.concat([x_pls, a_pls], axis=-1)
            f_pls = self.NOforward(t_pls, xa_pls, par_pls)
            x_pls_1 = self.sde.euler_onestep(t_now, x_now, dw1, par_now)
            xa_pls_1 = tf.concat([x_pls_1, a_pls], axis=-1)
            f_pls_1 = self.NOforward(t_pls, xa_pls_1, par_pls)
            x_pls_2 = self.sde.euler_onestep(t_now, x_now, dw2, par_now)
            xa_pls_2 = tf.concat([x_pls_2, a_pls], axis=-1)
            f_pls_2 = self.NOforward(t_pls, xa_pls_2, par_pls)
            f_hat_1 = f_now - self.h_tf(t_now, x_now, f_now, par_now) * self.dt + self.z_tf(t_now, x_now, grad, dw1, par_now)
            f_hat_2 = f_now - self.h_tf(t_now, x_now, f_now, par_now) * self.dt + self.z_tf(t_now, x_now, grad, dw2, par_now)
            loss_int = self.alpha * tf.reduce_mean((f_hat_1 - f_pls_1) * (f_hat_2 - f_pls_2))
        
        if self.config.initial_mode != 'random':
            loss_int += tf.reduce_mean(tf.math.reduce_variance(f_now[:, :, 0, :], 1))
        loss_tml = tf.square(f_pls[:, :, -1, :] - self.g_tf(x, param))
        loss = loss_int + loss_tml
        return f_now, loss, loss_int, loss_tml
    
    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor]) -> dict:
        with tf.GradientTape() as tape:
            with tf.name_scope('calling_model'):
                y, loss, loss_int, loss_tml = self(inputs[0])
            loss, loss_int, loss_tml = tf.reduce_mean(loss), tf.reduce_mean(loss_int), tf.reduce_mean(loss_tml)
            grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        y_true = inputs[1]
        error = tf.reduce_mean(tf.math.abs(y_true - y))
        return {"loss": loss,
                 "loss interior": loss_int,
                 "loss terminal": loss_tml,
                 "price error": error}
    
    def predict_price(self, x: tf.Tensor, params_eval: tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """
        Evaluate the model with input of time stamps and paths:
        x: asset path [B, M, T, d]
        params_eval: parameter needed to be inputted
        return: evaluated price tensor, analytical price tensor, time stamps for plotting
        """
        t = tf.expand_dims(self.time_stamp[0, :, :, :], axis=0)
        batch = tf.shape(x)[0]
        sample = tf.shape(x)[1]
        batch_num = batch * sample
        steps = tf.shape(x)[2]
        t_now = t[:, :, :-1, :]
        x_now = x[:, :, :-1, :]
        params_eval =  tf.reshape(params_eval, [1, 1, 1, 3])
        params_eval = tf.tile(params_eval, [1, self.config.M, self.config.time_steps, 1])
        y_true = self.option.exact_price(t, x, params_eval)
        sequence_x_t = tf.reshape(tf.concat([t_now, x_now], axis=-1), [batch_num, steps-1, 1+self.dim])
        a = tf.zeros([batch_num, self.n_a])  # (B*M, hidden)
        c = tf.zeros([batch_num, self.n_a])
        a_hidden = self.RNNCells(sequence_x_t, initial_state=[a, c]) 
        a = tf.expand_dims(a, 1)  # (batch_num, 1, hidden)
        a_hidden_1 = tf.reshape(tf.concat([a, a_hidden[:, :, :]], axis=1), [batch, sample, steps, -1]) 
        y_true = self.option.exact_price(t, x, params_eval)
        ax = tf.concat([x, a_hidden_1], axis=-1)
        y_pred = self.NOforward(t, ax, params_eval)
        return y_pred, y_true, t
    
    def h_tf(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, param: tf.Tensor) -> tf.Tensor: #get h function
        r = tf.expand_dims(param[:, :, :, 0], -1)
        return r * y

    def z_tf(self, t: tf.Tensor, x: tf.Tensor, grad: tf.Tensor, dw: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        v = tf.expand_dims(param[:, :, :, 1], -1)
        z = v * x * tf.reduce_sum(grad * dw, axis=-1, keepdims=True)
        return z

    def g_tf(self, x: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        payoff = self.option.payoff(x, param)
        return payoff

class BSDENonMarkovianModel:
    def __init__(self,sde, option, config):
        self.sde = sde
        self.config = config
        self.option = option

    def train(self, nr_epochs: int):
        learning_rate = self.config.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=200,
            decay_rate=0.9
        )
        Optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
        self.model = MarkovianSolver(self.sde, self.option, self.config)
        data_generator = DiffusionModelGenerator(self.sde, self.config, self.option)
        self.model.compile(optimizer=Optimizer)
        self.model.fit(x=data_generator, epochs=nr_epochs)

    def save_asset_price_series(self):
        seed = 600
        tf.random.set_seed(seed)
        np.random.seed(seed)
        r = np.random.uniform(0.038, 0.052)
        s = np.random.uniform(0.2, 0.6)
        m = np.random.normal(1.0, 0.1)
        params_eval = tf.constant([[r, s, m]])
        x ,_ ,_ ,_ = self.sde.sde_simulation(params_eval, self.config.M)
        y_pred, y_true, t = self.model.predict_price(x, params_eval)
        y_true = tf.squeeze(y_true).numpy()
        y_pred = tf.squeeze(y_pred).numpy()
        t_test = tf.squeeze(t).numpy()
        np.save('y_true_path.npy', y_true)
        np.save('y_pred_path.npy', y_pred)
        np.save('t.npy', t_test)
        import datetime
        now = datetime.datetime.now()
        print(f'paths are saved at {now}')


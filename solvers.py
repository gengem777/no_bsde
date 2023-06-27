import numpy as np
import tensorflow as tf
from abc import ABC
from function_space import DeepONet, DeepKernelONetwithPI, DeepKernelONetwithoutPI
from longstaff_solver import LongStaffSolver
from sde import HestonModel, TimeDependentGBM
from typing import List, Tuple

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
        if self.net_config.pi == "true":
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
        else:
            self.no_net = DeepKernelONetwithoutPI(branch_layer=self.branch_layers, 
                                                    trunk_layer=self.trunk_layers, 
                                                    dense=True, 
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
    
    def net_delta(self, inputs: Tuple[tf.Tensor]) -> tf.Tensor:
        t, x, u = inputs
        u_c, u_p = self.sde.split_uhat(u)
        y = self.no_net((t, x, u_c, u_p))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            y = self.no_net((t, x, u_c, u_p))
            delta = tape.gradient(y, x)
        if not isinstance(self.sde, HestonModel):
            z = delta[...,:self.dim]
        else:
            z = delta[...,:2 * self.dim]
        return delta[...,:self.dim]
        
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
    

    def h_tf(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:  # get h function
        """
        the driver term is r*y for MTG property
        """
        if not isinstance(self.sde, TimeDependentGBM):
            r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        else:
            r = self.sde.drift_onestep(t, x, u_hat)
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
    
    def z_hedge(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        grad = self.net_delta((t, x, u_hat))
        if not isinstance(self.sde, HestonModel):
            x = x[...,:self.dim]
            grad = grad[...,:self.dim]
            v_tx = self.sde.diffusion_onestep(t, x[...,:self.dim], u_hat)
        else:
            v_tx = self.sde.diffusion_onestep(t, x, u_hat)
        z = tf.reduce_sum(v_tx * grad, axis=-1, keepdims=True)
        return z

    def g_tf(self, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        payoff = self.option.payoff(x, u_hat)
        return payoff
    

class BermudanSolver(MarkovianSolver):
    """
    This is the class to construct the tain step of the BSDE loss function, the data is generated from three
    external classes: the sde class which yield the data of input parameters. the option class which yields the
    input function data and give the payoff function and exact option price.
    """

    def __init__(self, sde, option, config):
        super(BermudanSolver, self).__init__(sde, option, config)
        self.exer_index = self.option.exer_index
        self.num_tasks = 0

    def reset_task(self):
        self.num_tasks = 0

    def call(self, data: Tuple[tf.Tensor], training=None) -> Tuple[tf.Tensor]:
        """
        :param t: time B x M x (T-1) x 1
        :param x: state B x M x (T-1) x D
        :param  param: B x K ->(repeat) B x M x (T-1) x K
        :return: value (B, M, (T-1), 1), gradient_x (B, M, (T-1), D)
        """
        t, x, dw, u = data
        num_steps = self.eqn_config.time_steps
        y = self.g_tf(x, u) # (B, M, 1)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            f = self.net_forward((t, x, u))
            grad = tape.gradient(f, x) # grad wrt whole Markovian variable (X_t, M_t) 
                                            # First self.dim entry is the grad w.r.t X_t
        z = self.z_tf(t[:,:,:-1], x[:,:,:-1], grad[:,:,:-1], dw, u[:,:,:-1]) # (B, M, N, 1)
        y_values = tf.TensorArray(tf.float32, size=self.eqn_config.time_steps)
        y_values = y_values.write(num_steps-1, y)
        for n in reversed(range(num_steps - 1)):
            y = y - self.h_tf(t[:,:,n+1,:], x[:,:,n+1,:], y, u[:,:,n+1,:]) - z[:,:,n,:]
            if n in self.exer_index:
                if self.num_tasks == 0:
                    df = self.discount_factor(t[:,:,n,:], u[:,:,n,:])
                    y = df * self.g_tf(x, u)
                    y = tf.maximum(y, self.option.early_payoff(x[:,:,n,:], u[:,:,n,:]))
                else:
                    y = tf.maximum(y, self.option.early_payoff(x[:,:,n,:], u[:,:,n,:]))
            y_values = y_values.write(n, y)
        y_values = y_values.stack()
        y_values = tf.transpose(y_values, perm=[1, 2, 0, 3])
        # loss_variance = tf.reduce_mean(tf.math.reduce_variance(y, axis=1))
        loss_interior = tf.reduce_mean(tf.reduce_sum((f - y_values)**2, axis=2))
        #loss = self.alpha * loss_variance + loss_interior
        loss = loss_interior
        loss_terminal = tf.reduce_mean((f[:,:,-1,:] - y_values[:,:,-1,:])**2)
        return loss, loss_interior, loss_terminal

    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor]) -> dict:
        with tf.GradientTape() as tape:
            with tf.name_scope('calling_model'):
                loss, loss_interior, loss_terminal = self(inputs[0])
            grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {"loss": loss,
                "loss interior": loss_interior,
                "loss tml": loss_terminal}
    
    def h_tf(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:  # get h function
        """
        the driver term is r*y for MTG property
        t: [B, M, 1]
        x: [B, M, d]
        u_hat: [B, M, k]
        """
        if not isinstance(self.sde, TimeDependentGBM):
            r = tf.expand_dims(u_hat[:, :, 0], -1) # [B, M, 1]
        else:
            t = tf.expand_dims(t, axis=2)
            x = tf.expand_dims(x, axis=2)
            u_hat = tf.expand_dims(u_hat, axis=2)
            r = self.sde.drift_onestep(t, x, u_hat) # [B, M, 1]
            r = tf.squeeze(r, axis=2)
        return r * y
    
    def discount_factor(self, t: tf.Tensor, u_hat: tf.Tensor):
        T = self.eqn_config.T
        r = tf.expand_dims(u_hat[:, :, 0], -1)
        df = tf.exp(-r * (T-t))
        return df

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



class BermudanTwoStepSolver(MarkovianSolver): 
    def __init__(self, sde, option, config): 
        super(BermudanTwoStepSolver, self).__init__(sde, option, config)
        self.exer_index = self.option.exer_index
        solver = LongStaffSolver(sde, option, config)
        tf.config.run_functions_eagerly(False)
        print("first step training")
        solver.train()
        print("first step training end, we have value operators")
        self.subnets = solver.models
        self.exercise_index = self.option.exer_index

    def loss_subinterval(self, data: Tuple[tf.Tensor], idx: int):
        """
        This method calculate the European loss in a subinterval of two consecutive early-exercise dates
        the idx is the idx of the early-exercise date. when idx attain its maximum value, this reduces to the European loss whose
        terminal payoff is just exercise_value; Otherwise, this is the European loss with terminal payoff of max(continuation_value, exercise_value)
        """
        t, x, dw, u = data
        num_steps = tf.shape(x)[2]
        loss_interior = 0.0
        u_now = u[:, :, :-1, :]
        u_pls = u[:, :, 1:, :]
        x_now = x[:, :, :-1, :]
        x_pls = x[:, :, 1:, :]
        t_now = t[:, :, :-1, :]
        t_pls = t[:, :, 1:, :]
        if idx == len(self.subnets):
            target = self.g_tf(x, u)
        else:
            target = tf.maximum(self.g_tf(x, u), tf.stop_gradient(self.subnets[idx](x[:,:,-1,:], u[:,:,-1,:])))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_now)
            f_now = self.net_forward((t_now, x_now, u_now))
            grad = tape.gradient(f_now, x_now) # grad wrt whole Markovian variable (X_t, M_t) 
                                            # First self.dim entry is the grad w.r.t X_t
        f_pls = self.net_forward((t_pls, x_pls, u_pls))
        for n in range(num_steps-1):
            V_pls = f_pls[:, :, n:, :]
            V_now = f_now[:, :, n:, :]
            V_hat = V_now - self.h_tf(t_now[:,:, n:,:], x_now[:,:, n:,:], V_now, u_now[:,:, n:,:]) * self.dt + self.z_tf(t_now[:,:, n:,:], x_now[:,:, n:,:], 
                                                                            grad[:,:, n:,:], dw[:,:, n:,:], u_now[:,:, n:,:])
            tele_sum = tf.reduce_sum(V_pls - V_hat, axis=2)
            loss_interior += tf.reduce_mean(tf.square(tele_sum + target - f_pls[:, :, -1, :]))
        loss_tml = tf.reduce_mean(tf.square(f_pls[:, :, -1, :] - target))
        loss = self.alpha * loss_tml + loss_interior
        return loss
    
    def call(self, data: Tuple[tf.Tensor], training=None) -> Tuple[tf.Tensor]:
        """
        :param t: time B x M x (T-1) x 1
        :param x: state B x M x (T-1) x D
        :param  param: B x K ->(repeat) B x M x (T-1) x K
        :return: value (B, M, (T-1), 1), gradient_x (B, M, (T-1), D)
        """
        t, x, dw, u = data
        loss = 0.0
        for idx in range(len(self.subnets) + 1):
            if idx == 0:
                t = t[:,:,:self.exercise_index[idx],:]
                x = x[:,:,:self.exercise_index[idx],:]
                u = u[:,:,:self.exercise_index[idx],:]
                dw = dw[:,:,:self.exercise_index[idx]-1,:]
            
            elif idx == len(self.subnets):
                t = t[:,:,self.exercise_index[idx-1]:,:]
                x = x[:,:,self.exercise_index[idx-1]:,:]
                u = u[:,:,self.exercise_index[idx-1]:,:]
                dw = dw[:,:,self.exercise_index[idx-1]-1,:]
            
            else:
                t = t[:,:,self.exercise_index[idx-1]:self.exercise_index[idx],:]
                x = x[:,:,self.exercise_index[idx-1]:self.exercise_index[idx],:]
                u = u[:,:,self.exercise_index[idx-1]:self.exercise_index[idx],:]
                dw = dw[:,:,self.exercise_index[idx-1]:self.exercise_index[idx]-1,:]
            
            sub_data = t, x, dw, u
            loss += self.loss_subinterval(sub_data, idx)
        return loss



    




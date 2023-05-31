from abc import ABC, abstractmethod
from typing import Optional
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple


class ItoProcessDriver(ABC):
    """Abstract class for Ito processes, themselves driven by another Ito process B(t).
    This class represents the solution to an Ito stochastic differential equation with jump diffusion of the form
    $dX_t = r * X_t dt + \sigma(X(t), t) X_t dB(t)
    1. do initialization of X_0
    2. one step forward simulation
    3. simulate the path and output path tensor with Brownian motion increment tensor
    The class ItoProcessDriven implements common methods for sampling from the solution to this SDE,
    """

    def __init__(self, config):  # TODO
        """
        config is the set of the hyper-parameters
        we give the notations:
        dim: dimension of assets
        T; time horizon of simulation
        dt: length of time increments
        N: number of time steps
        x_init: initial state value
        vol_init: inital stochastic volatility value
        """
        self.config = config.eqn_config
        self.val_config = config.val_config
        self.dim = self.config.dim
        self.initial_mode = self.config.initial_mode
        self.x_init = self.config.x_init
        self.vol_init = self.config.vol_init
        self.range_list = []
        self.val_range_list = []
    
    def get_batch_size(self, u_hat: tf.Tensor):
        return tf.shape(u_hat)[0]

    def initial_sampler(self, u_hat: tf.Tensor) -> tf.Tensor:
        """
        Initial sampling of the asset price
        """
        batch_size = self.get_batch_size(u_hat)
        dimension = self.config.dim
        samples = self.config.sample_size
        # if dimension == 1:
        dist = tfp.distributions.TruncatedNormal(loc=self.x_init,
                                                        scale=self.x_init * 0.1,
                                                        low=self.x_init* 0.05,
                                                        high=self.x_init * 3.0)
        if self.initial_mode == 'random':
            state = tf.reshape(dist.sample(batch_size * samples * dimension), [batch_size, samples, dimension])

        elif self.initial_mode == 'fixed':
            state = self.x_init * tf.ones(shape=(batch_size, samples, dimension))

        elif self.initial_mode == 'partial_fixed':
            state = tf.reshape(dist.sample(batch_size), [batch_size, 1, 1])
            state = tf.tile(state, [1, samples, dimension])

        # else:
        #     dist = tfp.distributions.Dirichlet([self.initial_value] * dimension)
        #     state = dist.sample([batch_size, samples]) * tf.cast(dimension, dtype=tf.float32)

        return state
    
    
    def drift(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        Computes the drift of this stochastic process.
        operator setting
        mu: (batch_size, 1)
        state: (batch_size, path_size, dim)
        mu * state on a batch
        return: batch_size, path_size, dim)
        """
        batch = tf.shape(u_hat)[0]
        mu = tf.reshape(u_hat[:, 0], [batch, 1, 1])
        return mu * state

    @abstractmethod
    def diffusion(self, state: tf.Tensor, time: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def diffusion_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor):
        r"""
        get \sigma(t,X) with shape (B,M,N,d)
        """
        raise NotImplementedError
    
    def drift_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor):
        """
        all inputs are [batch, sample, T, dim] like
        what we do is calculate the drift for the whole tensor
        """
        rate_tensor = tf.expand_dims(u_hat[..., 0], -1)
        return rate_tensor * state_tensor

    def corr_matrix(self, state: tf.Tensor, u_hat: tf.Tensor):
        """
        In base class the corr is just identity matrix
        """
        batch = tf.shape(u_hat)[0]
        samples = tf.shape(state)[1]
        corr = tf.eye(self.dim, batch_shape=[batch, samples])
        return corr 
    
    def brownian_motion(self, state: tf.Tensor, u_hat: tf.Tensor):
        """
        generate non iid Brownian Motion increments for a certain time step
        the time increment is dt, the way to calculate the dependent BM increment:
        1. calculate corr matrix with the function corr_matrix (at least in this class)
        2. do cholasky decomposition on corr matrix to a low diagnal matrix L
        3. multiply the matrix L with the iid BMs
        denote dim as the dimension of asset, the return shape is (batch, paths, dim), and
        (batch, paths, dim * 2) under stochastic vol cases.
        """
        dt = self.config.dt
        batch_size = tf.shape(u_hat)[0]
        samples = tf.shape(state)[1]
        actual_dim = tf.shape(state)[2] #in SV model actual_dim = 2* dim =/= dim
        state_asset = state[...,:self.dim]
        corr = self.corr_matrix(state_asset, u_hat)
        cholesky_matrix = tf.linalg.cholesky(corr)
        white_noise = tf.random.normal([batch_size, samples, actual_dim], mean=0.0, stddev=tf.sqrt(dt))
        state_noise =  tf.einsum('...ij,...j->...i', cholesky_matrix, white_noise[...,:self.dim])
        vol_noise = white_noise[...,self.dim:]
        return tf.concat([state_noise, vol_noise], axis=-1)
        
    
    def euler_maryama_step(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor):
        """

        :param state: tf.Tensor
            The current state of the process.
            Shape is [batch, samples, dim].
        :param time: tf.Tensor
            The current time of the process. A scalar.
        :param noise: tf.Tensor
            The noise of the driving process.
            Shape is [batch, samples, dim].
        :param timestep : tf.Tensor
            A scalar; dt, the amount of time into the future in which we are stepping.
        :return (state, time) : (tf.Tensor, tf.Tensor)
            If this Ito process is X(t), the return value is (X(t+dt), dBt, St_*YdNt).
        """
        # TODO: high dim BM with cov matrix needed to be elaborated

        dt = self.config.dt
        noise = self.brownian_motion(state, u_hat)
        drift = self.drift(time, state, u_hat)
        diffusion = self.diffusion(time, state, u_hat)
        increment = drift * dt + diffusion * noise
        # jump_diffusion = self.jump_diffusion(state, time)
        return state + increment, noise
    
    def sde_simulation(self, u_hat: tf.Tensor, samples: int):
        """
        the whole simulation process with each step iterated by method euler_maryama_step(),
        return is a tuple of tensors recording the paths of state and BM increments
        x: path of states, shape: (batch_size, path_size, num_timesteps, dim)
        dw: path of BM increments, shape: (batch_size, path_size, num_timesteps-1, dim)
        """
        stepsize = self.config.dt
        time_steps = self.config.time_steps
        state_process = tf.TensorArray(tf.float32, size=time_steps)
        brownian_increments = tf.TensorArray(tf.float32, size=time_steps - 1)
        state = self.initial_sampler(u_hat)

        time = tf.constant(0.0)
        state_process = state_process.write(0, state)
        current_index = 1
        while current_index < time_steps:
            state, dw = self.euler_maryama_step(time, state, u_hat)
            state_process = state_process.write(current_index, state)
            brownian_increments = brownian_increments.write(current_index - 1, dw)
            current_index += 1
            time += stepsize
        x = state_process.stack()
        dw = brownian_increments.stack()
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        dw = tf.transpose(dw, perm=[1, 2, 0, 3])
        return x, dw
    
    def sample_parameters(self, N=100, training=True):  # N is the number of batch size
        """
        sample paremeters for simulating SDE given the range of each parameter from the config file,
        given the number of parameters need to be samples as k:
        the return tensorshape is [batch_size, k]
        """
        num_params = int(N * self.config.batch_size)
        if training:
            num_params = int(N * self.config.batch_size)
            return tf.concat([
                tf.random.uniform([num_params, 1], minval=p[0], maxval=p[1]) for p in self.range_list
            ], axis=1)
        else:
            num_params = int(N * self.val_config.batch_size)
            return tf.concat([
                tf.random.uniform([num_params, 1], minval=p[0], maxval=p[1]) for p in self.val_range_list
            ], axis=1)

    def expand_batch_inputs_dim(self, u_hat: tf.Tensor):
        """
        In order for consistence between input shape of parameter tensor and input state tensor into the network
        we need to unify the dimension. This method is to expand the dimension of 2-d tensor to 4-d using tf.tile method

        input: the parmeter tensor with shape [batch_size, K], K is the dim of parameters
        output: the expanded parameter [batch_size, sample_size, time_steps, K]
        """
        u_hat = tf.reshape(u_hat, [u_hat.shape[0], 1, 1, u_hat.shape[-1]])
        u_hat = tf.tile(u_hat, [1, self.config.sample_size, self.config.time_steps, 1])
        return u_hat
    
    def split_uhat(self, u_hat: tf.Tensor):
        """
        This method is to calculate the rate and volatility curve given a batch of parameters
        For example, GBM case, the parameters sampled has dimension batch_size + (3). where K=3
        and batch_size = (B, M, N)
        Then this function calculate the curve function based on parameter $\mu$ and $\sigma$
        return $\mu(t)$ and $\sigma(t)$ on the given support grid and return the batch of moneyness K
        Then Then return is a tuple of two tensors: (u_curve, u_param)
        u_curve: batch_size + (time_steps, num_curves), u_param = batch_size + (1)
        """
        return u_hat, _

class GeometricBrownianMotion(ItoProcessDriver):
    """
    A subclass of ItoProcess, MertonJumpDiffusionProcess under Q measure, mainly for testing.
    This class implements a multivariate geometric Brownian motion process
    with with Merton jump diffusion

    the parameter is a batch of parameter sampled from a distribution
    whose shape is (num_batch, dim_param = 2) \mu and \sigma
    one parameter correspond to num_sample paths
    """

    def __init__(self,
                 config):
        super().__init__(config)
        self.config = config.eqn_config
        self.initial_mode = self.config.initial_mode
        self.x_init = self.config.x_init
        self.r_range = self.config.r_range
        self.s_range = self.config.s_range
        self.range_list = [self.r_range, self.s_range]
        self.val_range_list = [self.val_config.r_range, self.val_config.s_range]
        if self.config.dim != 1:
            self.rho_range = self.config.rho_range
            #if not self.config.iid:
            self.range_list.append(self.rho_range)
            self.val_range_list.append(self.val_config.rho_range)

    def diffusion(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        Computes the instantaneous diffusion of this stochastic process.

        :param state : tf.Tensor
            Contains samples from the stochastic process at a specific time.
            Shape is [samples, self.dimension].
        :param time : tf.Tensor
            The current time; a scalar.

        :return diffusion: tf.Tensor
            The return is essentially a list of instantaneous diffusion matrices
            for each sampled state input.
            It is a tensor of shape [samples, self.dimension, self.dimension].

        param_input (B, 1)
        state (B, M, D)
        return (B, M, D)
        """
        batch = u_hat.shape[0]
        samples = state.shape[1]
        sigma = tf.reshape(u_hat[:, 1], [batch, 1, 1])
        sigma = tf.linalg.diag(tf.tile(sigma, [1, samples, self.dim]))
        return tf.einsum('...ij,...j->...i', sigma, state)
        
    def corr_matrix(self, state: tf.Tensor, u_hat: tf.Tensor):
        batch = state.shape[0]
        samples = state.shape[1]
        if not self.dim == 1:
            rho = tf.reshape(u_hat[:,2], [batch, 1, 1, 1])
            rho_mat = tf.tile(rho, [1, samples, self.dim, self.dim])
            i_mat = tf.eye(self.dim, batch_shape=[batch, samples])
            rho_diag = tf.linalg.diag(rho_mat[...,0])
            corr = i_mat - rho_diag + rho_mat
        else:
            corr = super(GeometricBrownianMotion, self).corr_matrix(state, u_hat)
        return corr
    
    def diffusion_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor):
        """
        get \sigma(t,X) with shape (B,M,N,d)
        in GBM \sigma(t,X) = \sigma * X
        """
        vol_tensor = tf.expand_dims(u_hat[..., 1], -1)
        return vol_tensor * state_tensor

    def euler_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, dw: tf.Tensor, u_hat: tf.Tensor):
        # assert self.diffusion(time, x_path, param).shape == dw.shape
        r = tf.expand_dims(u_hat[..., 0], -1)
        v = tf.expand_dims(u_hat[..., 1], -1)
        state_tensor_after_step = state_tensor + r * state_tensor * self.config.dt + v * state_tensor * dw
        return state_tensor_after_step
    
    def split_uhat(self, u_hat: tf.Tensor):
        """
        GBM case, the parameters sampled has dimension batch_size + (3). where K=3
        and batch_size = (B, M, N)
        Then this function calculate the curve function based on parameter $\mu$ and $\sigma$
        return $\mu(t)$ and $\sigma(t)$ on the given support grid and return the batch of moneyness K
        Then Then return is a tuple of two tensors: (u_curve, u_param)
        u_curve: batch_size + (time_steps, num_curves), u_param = batch_size + (1)
        """
        # batch_shape = (u_hat.shape[0], u_hat.shape[1], u_hat.shape[2])
        B_0 = tf.shape(u_hat)[0]
        B_1 = tf.shape(u_hat)[1]
        B_2 = tf.shape(u_hat)[2]
        u_curve = tf.reshape(u_hat[...,:2], [B_0, B_1, B_2, 1, 2])
        u_curve = tf.tile(u_curve, [1, 1, 1, self.config.sensors, 1])
        u_param = u_hat[..., 2:]
        return u_curve, u_param


class TimeDependentGBM(GeometricBrownianMotion):
    def __init__(self,
                 config):
        super().__init__(config)
        self.config = config.eqn_config
        self.initial_mode = self.config.initial_mode
        self.x_init = self.config.x_init
        self.r0_range = self.config.r0_range
        self.r1_range = self.config.r1_range
        self.r2_range = self.config.r2_range
        self.s0_range = self.config.s0_range
        self.beta_range = self.config.beta_range
        self.range_list = [self.r0_range, self.r1_range, self.r2_range, self.s0_range, self.beta_range]
        self.val_range_list = [self.val_config.r0_range, self.val_config.r1_range, self.val_config.r2_range, \
                               self.val_config.s0_range, self.val_config.beta_range]
        if self.config.dim != 1:
            self.rho_range = self.config.rho_range
            #if not self.config.iid:
            self.range_list.append(self.rho_range)
            self.val_range_list.append(self.val_config.rho_range)
        
        self.t_grid = tf.linspace(0., self.config.T, self.config.sensors)

    def drift(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        Computes the drift of this stochastic process.
        operator setting
        mu: (batch_size, 1)
        state: (batch_size, path_size, dim)
        mu * state on a batch
        return: batch_size, path_size, dim)
        """
        batch = tf.shape(u_hat)[0]
        r0 = tf.reshape(u_hat[:, 0], [batch, 1, 1])
        r1 = tf.reshape(u_hat[:, 1], [batch, 1, 1])
        r2 = tf.reshape(u_hat[:, 2], [batch, 1, 1])
        r = r0 + r1 * time + r2 * time ** 2 
        return r * state

    def diffusion(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        Computes the instantaneous diffusion of this stochastic process.

        :param state : tf.Tensor
            Contains samples from the stochastic process at a specific time.
            Shape is [samples, self.dimension].
        :param time : tf.Tensor
            The current time; a scalar.

        :return diffusion: tf.Tensor
            The return is essentially a list of instantaneous diffusion matrices
            for each sampled state input.
            It is a tensor of shape [samples, self.dimension, self.dimension].

        param_input (B, 1)
        state (B, M, D)
        return (B, M, D)
        """
        T = self.config.T
        batch = u_hat.shape[0]
        samples = state.shape[1]
        s_0 = tf.reshape(u_hat[:, 3], [batch, 1, 1])
        beta = tf.reshape(u_hat[:, 4], [batch, 1, 1])
        sigma = s_0 * tf.exp(beta * (time - T))
        sigma = tf.linalg.diag(tf.tile(sigma, [1, samples, self.dim]))
        return tf.einsum('...ij,...j->...i', sigma, state)
        
    # def corr_matrix(self, state: tf.Tensor, u_hat: tf.Tensor):
    #     batch = state.shape[0]
    #     samples = state.shape[1]
    #     if not self.dim == 1:
    #         rho = tf.reshape(u_hat[:,5], [batch, 1, 1, 1])
    #         rho_mat = tf.tile(rho, [1, samples, self.dim, self.dim])
    #         i_mat = tf.eye(self.dim, batch_shape=[batch, samples])
    #         rho_diag = tf.linalg.diag(rho_mat[...,0])
    #         corr = i_mat - rho_diag + rho_mat
    #     else:
    #         corr = super(TimeDependentGBM, self).corr_matrix(state, u_hat)
    #     return corr
    
    def drift_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor):
        r0 = tf.expand_dims(u_hat[..., 0], -1)
        r1 = tf.expand_dims(u_hat[..., 1], -1)
        r2 = tf.expand_dims(u_hat[..., 2], -1)
        r_t = r0 + r1 * time_tensor + r2 * time_tensor ** 2
        return r_t


    def diffusion_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor):
        """
        get \sigma(t,X) with shape (B,M,N,d)
        in GBM \sigma(t,X) = \sigma * X
        """
        T = self.config.T
        s_0 = tf.expand_dims(u_hat[..., 3], -1)
        beta = tf.expand_dims(u_hat[..., 4], -1)
        vol_tensor = s_0 * tf.exp(beta * (time_tensor - T))
        return vol_tensor * state_tensor

    def euler_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, dw: tf.Tensor, u_hat: tf.Tensor):
        # assert self.diffusion(time, x_path, param).shape == dw.shape
        r = self.drift_onestep(time_tensor, state_tensor, u_hat)
        vs = self.diffusion_onestep(time_tensor, state_tensor, u_hat)
        state_tensor_after_step = state_tensor + r * state_tensor * self.config.dt + vs * dw
        return state_tensor_after_step
    
    def split_uhat(self, u_hat: tf.Tensor):
        """
        GBM case, the parameters sampled has dimension batch_size + (3). where K=3
        and batch_size = (B, M, N)
        Then this function calculate the curve function based on parameter $\mu$ and $\sigma$
        return $\mu(t)$ and $\sigma(t)$ on the given support grid and return the batch of moneyness K
        Then Then return is a tuple of two tensors: (u_curve, u_param)
        u_curve: batch_size + (time_steps, num_curves), u_param = batch_size + (1)
        """
        # batch_shape = (u_hat.shape[0], u_hat.shape[1], u_hat.shape[2])
        t = tf.reshape(self.t_grid, [1, 1, 1, self.config.sensors, 1])
        B_0 = tf.shape(u_hat)[0]
        B_1 = tf.shape(u_hat)[1]
        B_2 = tf.shape(u_hat)[2]
        t = tf.tile(t, [B_0, B_1, B_2, 1, 1])
        r0 = tf.reshape(u_hat[...,0], [B_0, B_1, B_2, 1, 1])
        r0 = tf.tile(r0, [1, 1, 1, self.config.sensors, 1])
        r1 = tf.reshape(u_hat[...,1], [B_0, B_1, B_2, 1, 1])
        r1 = tf.tile(r1, [1, 1, 1, self.config.sensors, 1])
        r2 = tf.reshape(u_hat[...,2], [B_0, B_1, B_2, 1, 1])
        r2 = tf.tile(r2, [1, 1, 1, self.config.sensors, 1])
        r_curve = r0 + r1 * t * r2 * t ** 2
        
        T = self.config.T
        s0 = tf.reshape(u_hat[...,3], [B_0, B_1, B_2, 1, 1])
        s0 = tf.tile(s0, [1, 1, 1, self.config.sensors, 1])
        beta = tf.reshape(u_hat[...,4], [B_0, B_1, B_2, 1, 1])
        beta = tf.tile(beta, [1, 1, 1, self.config.sensors, 1])
        s_curve = s0 * tf.exp(beta * (t - T))

        u_curve = tf.concat([r_curve, s_curve], axis=-1)
        u_param = u_hat[..., 4:]
        return u_curve, u_param


class CEVModel(ItoProcessDriver):
    """
    A subclass of ItoProcess,  yield curve modeled by Nelson and Siegel and local volatility modeled by
    Cox and Ross we have
    """

    def __init__(self,
                 config):
        super().__init__(config)
        # self.beta0_range = [0.02, 0.06]
        # self.beta1_range = [0.02, 0.05]
        # self.beta2_range = [0.03, 0.05]
        self.r_range = [0.01, 0.1]
        self.sigma_range = [0.02, 0.05]
        self.gamma_range = [0.3, 1.2]
        self.range_list = [self.r_range, self.sigma_range, self.gamma_range]

    def vol_surf(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        time is with same shape as state [batch, sample, dim]
        param is [batch, 2], sigma=param[:, 3], gamma = param[:, 4]
        calculate sigma(t, S) = sigma * S ^{(gamma-2)/2}
        return same shape as state_t [batch, sample, dim]
        """
        batch = tf.shape(u_hat)[0]
        sigma = tf.reshape(u_hat[:, 1], [batch, 1, 1])
        gamma = tf.reshape(u_hat[:, 2], [batch, 1, 1])
        vol = sigma * state ** (gamma - 1)
        return vol

    def diffusion(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        Computes the instantaneous diffusion of this stochastic process.

        :param state : tf.Tensor
            Contains samples from the stochastic process at a specific time.
            Shape is [samples, self.dimension].
        :param time : tf.Tensor
            The current time; a scalar.

        :return diffusion: tf.Tensor
            The return is essentially a list of instantaneous diffusion matrices
            for each sampled state input.
            It is a tensor of shape [samples, self.dimension, self.dimension].

        param_input (B, 1)
        state (B, M, D)
        return (B, M, D)
        """
        batch = u_hat.shape[0]
        assert batch == state.shape[0]
        vol = self.vol_surf(time, state, u_hat)
        return vol * state

    def drift_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor):
        """
        all inputs are [batch, sample, T, dim] like
        what we do is calculate the drift for the whole tensor
        """
        rate_tensor = tf.expand_dims(u_hat[..., 0], -1)
        return rate_tensor * state_tensor

    def diffusion_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor):
        """
        get \sigma(t,X) with shape (B,M,N,d)
        in CEV \sigma(t,X) = \sigma * X^(b - 1)
        """
        sigma = tf.expand_dims(u_hat[..., 1], -1)
        gamma = tf.expand_dims(u_hat[..., 2], -1)
        vol_tensor = sigma * state_tensor ** (gamma - 1)
        return vol_tensor * state_tensor

    def euler_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, dw: tf.Tensor, u_hat: tf.Tensor):
        # assert self.diffusion(time, x_path, param).shape == dw.shape
        r = self.drift_onestep(time_tensor, state_tensor, u_hat)
        v = self.diffusion_onestep(time_tensor, state_tensor, u_hat)
        state_tensor_after_step = state_tensor + r * state_tensor * self.config.dt + v * state_tensor * dw
        return state_tensor_after_step
    


class HestonModel(ItoProcessDriver):
    def __init__(self,
                 config):
        super().__init__(config)
        self.r_range = self.config.r_range
        self.kappa_range = self.config.kappa_range
        self.theta_range = self.config.theta_range
        self.sigma_range = self.config.sigma_range
        self.rho_range = self.config.rho_range
        self.range_list = [self.r_range, self.theta_range, self.kappa_range, self.sigma_range, self.rho_range]
        self.val_range_list = [self.val_config.r_range, self.val_config.theta_range, self.val_config.kappa_range,
                                self.val_config.sigma_range, self.val_config.rho_range]
        if self.dim > 1:
            self.rhos_range = self.config.rhos_range
            self.range_list.append(self.rhos_range)
            self.val_range_list.append(self.val_config.rhos_range)

    def initial_sampler(self, u_hat: tf.Tensor) -> tf.Tensor:
        initial_state = super().initial_sampler(u_hat)
        new_state = initial_state * 0.1
        initial_value = tf.concat([initial_state, new_state], axis=-1)
        return initial_value

    def drift(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        In Heston model state = (S_t, V_t) with dim 2 * d
        S = state[:dim]
        V = state[dim:]
        drift_asset = r * S_t
        drift_vol = k(b - v_t)
        """
        batch = u_hat.shape[0]
        mu = tf.reshape(u_hat[:, 0], [batch, 1, 1])
        asset_state = state[:, :, :self.dim]
        vol_state   = tf.math.abs(state[:, :, self.dim:])
        asset_drift = mu * asset_state
        kappa = tf.reshape(u_hat[:, 1], [batch, 1, 1])
        theta = tf.reshape(u_hat[:, 2], [batch, 1, 1])
        vol_drift = kappa * (theta - vol_state)
        return tf.concat([asset_drift, vol_drift], axis=-1)
    
    def diffusion(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        Computes the instantaneous diffusion of this stochastic process.
        param_input (B, 1)
        state (B, M, D + D)
        return (B, M, D + D)
        diff_asset = sqrt(v_t) * S_t
        diff_vol = vol_of_vol * sqrt(v_t)
        """
        batch = u_hat.shape[0]
        asset_state = state[:, :, :self.dim]
        vol_state   = tf.math.abs(state[:, :, self.dim:])
        sqrt_vol = tf.math.sqrt(vol_state)
        asset_diffusion = sqrt_vol * asset_state
        vol_of_vol = tf.reshape(u_hat[:, 3], [batch, 1, 1])
        vol_diffusion   = vol_of_vol * sqrt_vol
        return tf.concat([asset_diffusion, vol_diffusion], axis=-1)
    
    def corr_matrix_1d(self, state: tf.Tensor, u_hat: tf.Tensor):
        batch = state.shape[0]
        samples = state.shape[1]
        actual_dim = state.shape[2]  
        rho = tf.reshape(u_hat[:,4], [batch, 1, 1, 1])
        rho_mat = tf.tile(rho, [1, samples, actual_dim, actual_dim])
        i_mat = tf.eye(actual_dim, batch_shape=[batch, samples])
        rho_diag = tf.linalg.diag(rho_mat[...,0])
        corr = i_mat - rho_diag + rho_mat
        return corr
    
    def cholesky_matrix_nd(self, state: tf.Tensor, u_hat: tf.Tensor):
        batch = state.shape[0]
        samples = state.shape[1]
        rho_s = tf.reshape(u_hat[:,5], [batch, 1, 1, 1])
        rho_s_mat = tf.tile(rho_s, [1, samples, self.dim, self.dim])
        zeros_mat = tf.zeros([batch, samples, self.dim, self.dim])
        i_mat = tf.eye(self.dim, batch_shape=[batch, samples])
        rho_s_diag = tf.linalg.diag(rho_s_mat[...,0])
        corr_s = i_mat - rho_s_diag + rho_s_mat
        cholesky_s = tf.linalg.cholesky(corr_s)
        rho_sv = tf.reshape(u_hat[:,4], [batch, 1, 1, 1])
        rho_sv_mat = tf.tile(rho_sv, [1, samples, self.dim, self.dim])
        rho_sv_diag = tf.linalg.diag(rho_sv_mat[...,0])

        #concat block matrixs
        a = tf.concat([cholesky_s, rho_sv_diag], axis=3)
        b = tf.concat([zeros_mat, i_mat], axis=3)
        return tf.concat([a, b], axis=2)

    
    def brownian_motion(self, state: tf.Tensor, u_hat: tf.Tensor):
        dt = self.config.dt
        batch_size = tf.shape(u_hat)[0]
        samples = tf.shape(state)[1]
        actual_dim = tf.shape(state)[2] #in SV model actual_dim = 2* dim =/= dim
        if self.dim == 1:
            corr = self.corr_matrix_1d(state, u_hat)
            cholesky_matrix = tf.linalg.cholesky(corr)
        else:
            cholesky_matrix = self.cholesky_matrix_nd(state, u_hat)
        white_noise = tf.random.normal([batch_size, samples, actual_dim], mean=0.0, stddev=tf.sqrt(dt))
        state_noise =  tf.einsum('...ij,...j->...i', cholesky_matrix, white_noise)
        return state_noise
    
    def drift_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor):
        """
        all inputs are [batch, sample, T, dim * 2] like
        what we do is calculate the drift for the whole tensor
        output: (r * S_t , k(b - v_t)) for all t with shape [batch, sample, T, dim * 2]
        """
        assert state_tensor.shape[0] == u_hat.shape[0]
        rate_tensor = tf.expand_dims(u_hat[..., 0], -1)
        kappa_tensor = tf.expand_dims(u_hat[..., 1], -1)
        theta_tensor = tf.expand_dims(u_hat[..., 2], -1)
        s_tensor = state_tensor[..., :self.dim]
        v_tensor = state_tensor[..., self.dim:]
        drift_s = rate_tensor * s_tensor
        drift_v = kappa_tensor * (theta_tensor - v_tensor)
        return tf.concat([drift_s, drift_v], axis=-1)

    def diffusion_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor):
        """
        get \sigma(t,X) with shape (B,M,N,d)
        in Heston \sigma(t,X) = (\sqrt(V_t) * X_t, vol_of_vol * \sqrt(V_t))
        """
        assert state_tensor.shape[0] == u_hat.shape[0]
        vol_of_vol = tf.expand_dims(u_hat[..., 3], -1)
        s_tensor = state_tensor[..., :self.dim]
        v_tensor = state_tensor[..., self.dim:]
        sqrt_v = tf.math.sqrt(tf.math.abs(v_tensor))
        diff_s = sqrt_v * s_tensor
        diff_v = vol_of_vol * sqrt_v
        return tf.concat([diff_s, diff_v], axis=-1)
    
    def split_uhat(self, u_hat: tf.Tensor):
        r"""
        Heston case, the parameters sampled has dimension batch_size + (5). where K=5
        and batch_size = (B, M, N)
        Then this function calculate the curve function based on parameter $r$ and $\theta$
        return $r(t)$ and $\theta(t)$ on the given support grid and return the batch of moneyness (\kappa, \sigma, K)
        Then Then return is a tuple of two tensors: (u_curve, u_param)
        u_curve: batch_size + (time_steps, num_curves), u_param = batch_size + (3)
        """
        B_0 = tf.shape(u_hat)[0]
        B_1 = tf.shape(u_hat)[1]
        B_2 = tf.shape(u_hat)[2]
        u_curve = tf.reshape(u_hat[...,:2], [B_0, B_1, B_2, 1, 2])
        u_curve = tf.tile(u_curve, [1, 1, 1, self.config.sensors, 1])
        u_param = u_hat[..., 2:]
        return u_curve, u_param

        

    






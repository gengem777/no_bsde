from abc import ABC, abstractmethod
from typing import Optional
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple


class ItoProcessDriver(ABC):
    """Abstract class for Ito processes, themselves driven by another Ito process B(t).
    This class represents the solution to an Ito stochastic differential equation with jump diffusion of the form
    $dX(t) = \mu(X(t), t)dt + \sigma(X(t), t)dB(t) + \int \gamma(t, X(t), z) \hat{N}(dt, dz)$
    1. do initialization of X_0
    2. one step forward simulation
    3. simulate the path and output path tensor with Brownian motion increment tensor
    The class ItoProcessDriven implements common methods for sampling from the solution to this SDE,
    """

    def __init__(self, config):  # TODO
        """
        config is the set of the hyper-parameters
        """
        self.config = config
        self.dim = self.config.dim

    @abstractmethod
    def initial_sampler(self, *args, **kwargs) -> tf.Tensor:
        pass

    @abstractmethod
    def drift(self, state: tf.Tensor, time: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def diffusion(self, state: tf.Tensor, time: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def euler_maryama_step(self, state_t: tf.Tensor, param: tf.Tensor):
        pass

    @abstractmethod
    def sde_simulation(self, *args, **kwargs):
        pass


class GeometricBrowianMotion(ItoProcessDriver):
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
        self.initial_mode = config.initial_mode
        self.x_init = self.config.x_init
        self.r_range = [0.01, 0.1]
        self.sigma_range = [0.02, 0.8]

    def get_batch_size(self, param: tf.Tensor):
        return tf.shape(param)[0]

    # @tf.function
    def initial_sampler(self, param: tf.Tensor) -> tf.Tensor:
        batch_size = self.get_batch_size(param)
        dimension = self.config.dim
        samples = self.config.M
        if dimension == 1:
            if self.initial_mode == 'random':
                dist = tfp.distributions.TruncatedNormal(loc=self.initial_value * 1.5,
                                                         scale=self.initial_value * 0.2,
                                                         low=0.5,
                                                         high=3.0)
                state = tf.reshape(dist.sample(batch_size * samples * dimension), [batch_size, samples, dimension])

            elif self.initial_mode == 'fixed':
                state = self.initial_value * tf.ones(shape=(batch_size, samples, dimension))

            elif self.initial_mode == 'partial_fixed':
                dist = tfp.distributions.TruncatedNormal(loc=self.initial_value * 1.5,
                                                         scale=self.initial_value * 0.2,
                                                         low=0.5,
                                                         high=3.0)
                state = tf.reshape(dist.sample(batch_size * dimension), [batch_size, 1, dimension])
                state = tf.tile(state, [1, samples, 1])

        else:
            dist = tfp.distributions.Dirichlet([1.0] * dimension)
            state = dist.sample([batch_size, samples]) * tf.cast(dimension, dtype=tf.float32)

        return state

    def drift(self, time: tf.Tensor, state: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        """
        Computes the drift of this stochastic process.

        :param state : tf.Tensor
            Contains samples from the stochastic process at a specific time.
            Shape is [self.samples, self.dimension].
        :param time : tf.Tensor
            The time in question; a scalar,
        under Q measure drift = r - q
        :return drift: tf.Tensor
            Tensor is a list of instantaneous drifts for each sampled state input,
            a tensor of shape [self.samples, self.dimension].

        operator setting
        mu: (B, 1)
        state: (B, M, D)
        mu * state on a batch
        return: (B, M, D)
        """
        batch = param.shape[0]
        assert batch == state.shape[0]
        if len(state.shape) == 3:
            mu = tf.reshape(param[:, 0], [batch, 1, 1])
        elif len(state.shape) == 4:
            mu = tf.reshape(param[:, 0], [batch, 1, 1, 1])
        assert len(state.shape) == len(state.shape)
        return mu * state

    def diffusion(self, time: tf.Tensor, state: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
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
        batch = param.shape[0]
        assert batch == state.shape[0]
        if len(state.shape) == 3:
            sigma = tf.reshape(param[:, 1], [batch, 1, 1])
        elif len(state.shape) == 4:
            sigma = tf.reshape(param[:, 1], [batch, 1, 1, 1])
        assert state.ndim == sigma.ndim
        return sigma * state

    def euler_maryama_step(self, time: tf.Tensor, state: tf.Tensor, param: tf.Tensor) -> (tf.Tensor, tf.Tensor):
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
        batch_size = tf.shape(param)[0]
        samples = tf.shape(state)[1]
        dim = tf.shape(state)[2]
        # noise = tf.squeeze(tf.random.normal([Batch_size, samples, self.noise_dimension], mean=0.0, stddev=tf.sqrt(stepsize)))
        noise = tf.random.normal([batch_size, samples, dim], mean=0.0, stddev=tf.sqrt(dt))

        drift = self.drift(time, state, param)
        diffusion = self.diffusion(time, state, param)
        # increment = drift*stepsize + tf.einsum("smn,sn->sm", diffusion, noise)
        assert diffusion.shape == noise.shape
        increment = drift * dt + diffusion * noise
        # jump_diffusion = self.jump_diffusion(state, time)
        return state + increment, noise

    def euler_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, dw: tf.Tensor, param: tf.Tensor):
        # assert self.diffusion(time, x_path, param).shape == dw.shape
        r = tf.expand_dims(param[..., 0], -1)
        v = tf.expand_dims(param[..., 1], -1)
        state_tensor_after_step = state_tensor + r * state_tensor * self.config.dt + v * state_tensor * dw
        return state_tensor_after_step

    def sde_simulation(self, param: tf.Tensor, samples: int) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
        """
        the whole simulation process
        """
        stepsize = self.config.dt
        time_steps = self.config.time_steps
        dimension = self.config.dim
        state_process = tf.TensorArray(tf.float32, size=time_steps)
        brownian_increments = tf.TensorArray(tf.float32, size=time_steps - 1)
        batch_size = tf.shape(param)[0]
        state = self.initial_sampler(param)

        time = tf.constant(0.0)
        state_process = state_process.write(0, state)
        current_index = 1
        while current_index < time_steps:
            state, dw = self.euler_maryama_step(time, state, param)
            state_process = state_process.write(current_index, state)
            brownian_increments = brownian_increments.write(current_index - 1, dw)
            current_index += 1
            time += stepsize
        x = state_process.stack()
        dw = brownian_increments.stack()
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        dw = tf.transpose(dw, perm=[1, 2, 0, 3])
        dw1 = tf.random.normal([batch_size, samples, time_steps - 1, dimension], mean=0.0, stddev=tf.sqrt(stepsize))
        dw2 = tf.random.normal([batch_size, samples, time_steps - 1, dimension], mean=0.0, stddev=tf.sqrt(stepsize))
        return x, dw, dw1, dw2

    def sample_parameters(self, N=100):  # N is the time of batch size
        r_range = self.r_range
        sigma_range = self.sigma_range
        num_params = N * self.config.batch_size
        r = tf.random.uniform([num_params, 1], minval=r_range[0], maxval=r_range[1])
        s = tf.random.uniform([num_params, 1], minval=sigma_range[0], maxval=sigma_range[1])
        params = tf.concat([r, s], 1)
        return params

    def expand_batch_inputs_dim(self, par: tf.Tensor):
        """
        input: the parmeter tensor with shape [batch_size, K], K is the dim of parameters
        output: the expanded parameter [batch_size, sample_size, time_steps, K]
        """
        par = tf.reshape(par, [self.config.batch_size, 1, 1, 2])
        par = tf.tile(par, [1, self.config.M, self.config.time_steps, 1])
        return par

    @property
    def initial_value(self):
        """
        A simple getter for the initial value.
        """
        return self.config.x_init


class LocalVolModel(ItoProcessDriver):
    """
    A subclass of ItoProcess,  yield curve modeled by Nelson and Siegel and local volatility modeled by
    Cox and Ross we have
    """

    def __init__(self,
                 config):
        super().__init__(config)
        self.initial_mode = config.initial_mode
        self.x_init = self.config.x_init

        self.beta0_range = [0.02, 0.06]
        self.beta1_range = [0.02, 0.05]
        self.beta2_range = [0.03, 0.05]
        self.sigma_range = [0.02, 0.80]
        self.gamma_range = [0.01, 2.10]

    def get_batch_size(self, param: tf.Tensor):
        return tf.shape(param)[0]

    def initial_sampler(self, param: tf.Tensor) -> tf.Tensor:
        batch_size = self.get_batch_size(param)
        dimension = self.config.dim
        samples = self.config.M
        if dimension == 1:
            if self.initial_mode == 'random':
                dist = tfp.distributions.TruncatedNormal(loc=self.initial_value * 1.5,
                                                         scale=self.initial_value * 0.2,
                                                         low=0.5,
                                                         high=3.0)
                state = tf.reshape(dist.sample(batch_size * samples * dimension), [batch_size, samples, dimension])

            elif self.initial_mode == 'fixed':
                state = self.initial_value * tf.ones(shape=(batch_size, samples, dimension))

            elif self.initial_mode == 'partial_fixed':
                dist = tfp.distributions.TruncatedNormal(loc=self.initial_value * 1.5,
                                                         scale=self.initial_value * 0.2,
                                                         low=0.5,
                                                         high=3.0)
                state = tf.reshape(dist.sample(batch_size * dimension), [batch_size, 1, dimension])
                state = tf.tile(state, [1, samples, 1])

        else:
            dist = tfp.distributions.Dirichlet([1.0] * dimension)
            state = dist.sample([batch_size, samples]) * tf.cast(dimension, dtype=tf.float32)

        return state

    def rate_curve(self, time: tf.Tensor, state: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        """
        time is with same shape as state [batch, sample, dim]
        param is [batch, 5], bata_i=param[:, i], i = 0,1,2
        calculate r(t) = \beta_0 + (\beta_1 + \beta_2 t)e^{-t}
        return same shape as state_t [batch, sample, dim]
        """
        tau = (self.config.T - time) * tf.ones_like(state)
        batch = tf.shape(param)[0]
        b0 = tf.reshape(param[:, 0], [batch, 1, 1])
        b1 = tf.reshape(param[:, 1], [batch, 1, 1])
        b2 = tf.reshape(param[:, 2], [batch, 1, 1])
        rate = b0 * tf.ones(tf.shape(tau)) + b1 * tf.math.exp(-tau) + b2 * tau * tf.math.exp(-tau)
        return rate

    def drift(self, time: tf.Tensor, state: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        """
        Computes the drift of this stochastic process.

        :param state : tf.Tensor
            Contains samples from the stochastic process at a specific time.
            Shape is [self.samples, self.dimension].
        :param time : tf.Tensor
            The time in question; a scalar,
        under Q measure drift = r(t)
        :return drift: tf.Tensor
            Tensor is a list of instantaneous drifts for each sampled state input,
            a tensor of shape [self.batch, self.samples, self.dim].

        operator setting
        mu: (B, 1)
        state: (B, M, D)
        mu * state on a batch
        return: (B, M, D)
        """
        batch = param.shape[0]
        assert batch == state.shape[0]
        mu = self.rate_curve(time, state, param)
        return mu * state

    def vol_surf(self, time: tf.Tensor, state: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
        """
        time is with same shape as state [batch, sample, dim]
        param is [batch, 2], sigma=param[:, 3], gamma = param[:, 4]
        calculate sigma(t, S) = sigma * S ^{(gamma-2)/2}
        return same shape as state_t [batch, sample, dim]
        """
        batch = tf.shape(param)[0]
        sigma = tf.reshape(param[:, 3], [batch, 1, 1])
        gamma = tf.reshape(param[:, 4], [batch, 1, 1])
        vol = sigma * state ** ((gamma - 2) / 2)
        return vol

    def diffusion(self, time: tf.Tensor, state: tf.Tensor, param: tf.Tensor) -> tf.Tensor:
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
        batch = param.shape[0]
        assert batch == state.shape[0]
        vol = self.vol_surf(time, state, param)
        return vol * state

    def euler_maryama_step(self, time: tf.Tensor, state: tf.Tensor, param: tf.Tensor) -> (tf.Tensor, tf.Tensor):
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
        batch_size = tf.shape(param)[0]
        samples = tf.shape(state)[1]
        dim = tf.shape(state)[2]
        # noise = tf.squeeze(tf.random.normal([Batch_size, samples, self.noise_dimension], mean=0.0, stddev=tf.sqrt(stepsize)))
        noise = tf.random.normal([batch_size, samples, dim], mean=0.0, stddev=tf.sqrt(dt))

        drift = self.drift(time, state, param)
        diffusion = self.diffusion(time, state, param)
        # increment = drift*stepsize + tf.einsum("smn,sn->sm", diffusion, noise)
        assert diffusion.shape == noise.shape
        increment = drift * dt + diffusion * noise
        # jump_diffusion = self.jump_diffusion(state, time)
        return state + increment, noise

    def drift_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, param: tf.Tensor):
        """
        all inputs are [batch, sample, T, dim] like
        what we do is calculate the drift for the whole tensor
        """
        b0 = tf.expand_dims(param[..., 0], -1)
        b1 = tf.expand_dims(param[..., 1], -1)
        b2 = tf.expand_dims(param[..., 2], -1)
        rate_tensor = b0 + b1 * tf.math.exp(-time_tensor) + b2 * time_tensor * tf.math.exp(-time_tensor)
        return rate_tensor * state_tensor

    def diffusion_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, param: tf.Tensor):
        """
        all inputs are [batch, sample, T, dim] like
        what we do is calculate the diffusion for the whole tensor
        """
        sigma = tf.expand_dims(param[..., 3], -1)
        gamma = tf.expand_dims(param[..., 4], -1)
        vol_tensor = sigma * state_tensor ** ((gamma - 2) / 2)
        return vol_tensor * state_tensor

    def euler_onestep(self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, dw: tf.Tensor, param: tf.Tensor):
        # assert self.diffusion(time, x_path, param).shape == dw.shape
        r = self.drift_onestep(time_tensor, state_tensor, param)
        v = self.diffusion_onestep(time_tensor, state_tensor, param)
        state_tensor_after_step = state_tensor + r * state_tensor * self.config.dt + v * state_tensor * dw
        return state_tensor_after_step

    def sde_simulation(self, param: tf.Tensor, samples: int) -> (tf.Tensor, tf.Tensor, tf.Tensor):

        stepsize = self.config.dt
        time_steps = self.config.time_steps
        dimension = self.config.dim
        state_process = tf.TensorArray(tf.float32, size=time_steps)
        brownian_increments = tf.TensorArray(tf.float32, size=time_steps - 1)
        # jump_increments = tf.TensorArray(tf.float32, size=time_steps-1)
        batch_size = tf.shape(param)[0]
        state = self.initial_sampler(param)
        time = tf.constant(0.0)
        state_process = state_process.write(0, state)
        current_index = 1
        while current_index < time_steps:
            state, dw = self.euler_maryama_step(time, state, param)
            state_process = state_process.write(current_index, state)
            brownian_increments = brownian_increments.write(current_index - 1, dw)
            # jump_increments = jump_increments.write(current_index-1, dj)
            current_index += 1
            time += stepsize
        x = state_process.stack()
        dw = brownian_increments.stack()
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        dw = tf.transpose(dw, perm=[1, 2, 0, 3])
        dw1 = tf.random.normal([batch_size, samples, time_steps - 1, dimension], mean=0.0, stddev=tf.sqrt(stepsize))
        dw2 = tf.random.normal([batch_size, samples, time_steps - 1, dimension], mean=0.0, stddev=tf.sqrt(stepsize))
        return x, dw, dw1, dw2

    def sample_parameters(self, N=100):  # N is the time of batch size
        num_params = N * self.config.batch_size
        return tf.concat([
            tf.random.uniform([num_params, 1], minval=m[0], maxval=m[1]) for m in [
                self.beta0_range,
                self.beta1_range,
                self.beta2_range,
                self.sigma_range,
                self.gamma_range,
            ]
        ], axis=1)

    def expand_batch_inputs_dim(self, par: tf.Tensor):
        """
        input: the parmeter tensor with shape [batch_size, K], K is the dim of parameters
        output: the expanded parameter [batch_size, sample_size, time_steps, K]
        """
        par = tf.reshape(par, [self.config.batch_size, 1, 1, 5])
        par = tf.tile(par, [1, self.config.M, self.config.time_steps, 1])
        return par

    @property
    def initial_value(self):
        """
        A simple getter for the initial value.
        """
        return self.config.x_init


class HestonModel(ItoProcessDriver):
    pass


class MertonJumpModel(ItoProcessDriver):
    pass


class StochasticVolJump(ItoProcessDriver):
    pass

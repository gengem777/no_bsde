import tensorflow as tf
from abc import ABC, abstractmethod
from dataclasses import dataclass
import tensorflow_probability as tfp

tfd = tfp.distributions
dist = tfd.Normal(loc=0., scale=1.)


@dataclass
class BaseOption:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def exact_price(self, t: tf.Tensor, x: tf.Tensor, params):
        raise NotImplementedError
    
    def expand_batch_inputs_dim(self, par: tf.Tensor):
        """
        input: the parmeter tensor with shape [batch_size, 1]
        output: the expanded parameter [batch_size, sample_size, time_steps, 1]
        """
        par = tf.reshape(par, [self.config.batch_size, 1, 1, 1])
        par = tf.tile(par, [1, self.config.M, self.config.time_steps, 1])
        return par



class EuropeanOption(BaseOption):
    def __init__(self, config, call_put_flag=True):
        super(EuropeanOption, self).__init__(config)
        """
        Parameters
        ----------
        m: moneyness with distribution N(1, 0.2)
        """
        self.call_put_flag = call_put_flag
        self.m_range = [0.4, 2.0]

    def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        x: tf.Tensor
            Asset price path. Tensor of shape (batch_size, sample_size, time_steps, d)
        param: tf.Tensor
            The parameter vectors of brunch net input and the last entry is for strike.
        Returns
        -------
        payoff: tf.Tensor
            basket option payoff. Tensor of shape (batch_size, sample_size, 1)
            when self.config.dim = 1, this reduced to 1-d vanilla call payoff
        """
        k = tf.expand_dims(param[:, :, 0, -1], axis=-1) # (B, M, 1)
        temp = tf.reduce_mean(x[:, :, -1, :], axis=-1, keepdims=True) # (B, M, 1)
        if self.call_put_flag:
            return tf.nn.relu(temp - k)
        else:
            return tf.nn.relu(k - temp)

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, params: tf.Tensor):
        """
        Implement the BS formula
        """
        T = self.config.T
        r = tf.expand_dims(params[:, :, :, 0], -1)
        vol = tf.expand_dims(params[:, :, :, 1], -1)
        K = tf.expand_dims(params[:, :, :, 2], -1)
        d1 = (tf.math.log(x / K) + (r + vol ** 2 / 2) * (T - t)) / (vol * tf.math.sqrt(T - t) + 0.000001)
        d2 = d1 - vol * tf.math.sqrt(T - t)
        if self.call_put_flag == True:
            c = x * dist.cdf(d1) - K * tf.exp(-r * (T - t)) * dist.cdf(d2)
        else:
            c = K * tf.exp(-r * (T - t)) * dist.cdf(-d2) - x * dist.cdf(-d1)
        return c

    def sample_parameters(self, N=100):  # N is the time of batch size
        m_range = self.m_range
        num_params = N * self.config.batch_size
        m = tf.random.uniform([num_params, 1], minval=m_range[0], maxval=m_range[1])
        return m


class LookbackOption(EuropeanOption):
    def __init__(self, config):
        super(LookbackOption, self).__init__(config)

    def markovian_var(self, x: tf.Tensor):
        """
        x is a (B, M, N, d) size
        The output is the cummin on the axis=2
        """
        m_pre = x[:, :, 0, :]
        m_list = [m_pre]
        for i in range(1, tf.shape(x)[2]):
            m_pre = tf.math.minimum(m_pre, x[:, :, i, :])
            m_list.append(m_pre)
        markov = tf.stack(m_list, axis=2)
        return markov

    def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
        temp = tf.reduce_mean(x[:, :, -1, :], axis=-1, keepdims=True) # (B, M, 1)
        float_min = tf.math.reduce_min(x, axis=2, keepdims=True) # (B, M, 1, d)
        temp_min = tf.reduce_mean(float_min, axis=-1)
        return temp - temp_min

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, params: tf.Tensor):
        pass



# region
# class EuropeanOptionJump(EuropeanOption):
#     def __init__(self, config, call_put_flag=True):
#         super(EuropeanOptionJump, self).__init__(config, call_put_flag)

#     def exact_price(self, t, x, params):
#         """
#         Use Montecarlo to calculate the price at the initial time
#         The input parameter shape x: [B, M, T, d], params: [B, M, None, k]
#         output return y_0: [B, 1]
#         """
#         T = self.config.T

#         def inner_bs(t, x, r, v, k):
#             """
#             imagine they are dim-4
#             """

#         payoff = self.payoff(x, params)
#         mc_price = tf.math.exp(-r * T) * tf.reduce_mean(payoff, axis=1)
#         return mc_price


# class DownandOutCall(BaseOption):
#     def __init__(self, config):
#         self.config = config

#     def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
#         barrier = tf.reshape(param[:, -2], [self.config.batch_size, 1, 1])
#         barrier = tf.tile(barrier, [1, self.config.M, self.config.time_steps])
#         x_init = tf.expand_dims(x[:, :, 0, :], axis=2)
#         temp = tf.reduce_min(x/x_init, axis=2)
#         mask = tf.cast(temp < barrier, tf.float32)
#         temp_mean = tf.reduce_mean(x[:, :, -1, :], axis=-1, keepdims=True)
#         k = tf.reshape(param[:, -1],[self.config.batch_size, 1, 1])
#         k = tf.tile(k, [1, self.config.M, 1])
#         return tf.nn.relu(temp_mean - k) * mask
# endregion

class GeometricAsian(EuropeanOption):
    def __init__(self, config):
        """
        Parameters
        ----------
        K: float or torch.tensor
            Strike. Id K is a tensor, it needs to have shape (batch_size)
        """
        super(GeometricAsian, self).__init__(config)
    
    def markovian_var(self, x: tf.Tensor):
        """
        x is a (B, M, N, d) size
        The output is the running integral on the axis=2
        The geometric average is:

        $$
        G_t=\exp \left\{\frac{1}{t} \int_0^t \log x_u d u\right\}
        $$
        """
        dt = self.config.dt
        dt_tensor = tf.math.cumsum(tf.ones(tf.shape(x)) * dt, axis=2)
        sumlog = tf.math.cumsum(tf.math.log(x), axis=2)
        log_average = sumlog * dt / dt_tensor
        geo_average = tf.exp(log_average)
        return geo_average
   
    def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        x: tf.Tensor
            Asset price path. Tensor of shape (B, M, N, d)
        param: tf.Tensor
            The parameter vectors of NO input and the last entry is for strike.
        Returns
        -------
        payoff: tf.Tensor
            Asian option payoff. Tensor of shape (B, M, 1)
        """
        k = tf.expand_dims(param[:, :, 0, -1], axis=-1) # (B, M, 1)
        dt = self.config.dt
        T = self.config.T
        geo_mean = tf.reduce_mean(tf.exp(tf.reduce_sum(tf.math.log(x), axis=2) * dt / T), axis=-1, keepdims=True)
        return tf.nn.relu(geo_mean - k)


    def exact_price(self, t: tf.Tensor, x: tf.Tensor, params):
        pass

    




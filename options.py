import tensorflow as tf
from dataclasses import dataclass
import tensorflow_probability as tfp

tfd = tfp.distributions
dist = tfd.Normal(loc=0., scale=1.)


@dataclass
class BaseOption:
    def __init__(self, config):
        self.config = config

    def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
        raise NotImplementedError

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
        self.m_range = [1.0, 0.05]
        # self.m_range = [100, 5]

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
            c = self.config.x_init * (x * dist.cdf(d1) - K * tf.exp(-r * (T - t)) * dist.cdf(d2))
        else:
            c = self.config.x_init * (K * tf.exp(-r * (T - t)) * dist.cdf(-d2) - x * dist.cdf(-d1))
        return c

    def sample_parameters(self, N=100):  # N is the time of batch size
        m_range = self.m_range
        num_params = int(N * self.config.batch_size)
        m = tf.math.maximum(tf.random.normal([num_params, 1], m_range[0], m_range[1]), 0.9)
        return m


class EuropeanBasketOption(EuropeanOption):
    def __init__(self, config):
        super(EuropeanBasketOption, self).__init__(config)
    
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
        temp = tf.exp(tf.reduce_mean(tf.math.log(x[:, :, -1, :]), axis=-1, keepdims=True))# (B, M, 1)
        if self.call_put_flag:
            return tf.nn.relu(temp - k)
        else:
            return tf.nn.relu(k - temp)

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        """
        Implement the BS formula
        """
        d = self.config.dim
        T = self.config.T
        r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
        rho = tf.expand_dims(u_hat[:, :, :, 2], -1)
        k = tf.expand_dims(u_hat[:, :, :, 3], -1)
        vol_bar = vol * tf.math.sqrt(1/d + rho * (1-1/d))
        S_pi = tf.exp(tf.reduce_mean(tf.math.log(x), axis=-1, keepdims=True))
        F = S_pi * tf.exp((r - vol**2/2 + vol_bar**2/2) * (T - t))
        d_1 = (tf.math.log(F/k) + vol_bar ** 2/2 * (T - t))/vol_bar/tf.sqrt(T - t)
        d_2 = d_1 - vol_bar * tf.sqrt(T - t)
        c = F * dist.cdf(d_1) - k * dist.cdf(d_2)
        return c




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

    def payoff(self, x: tf.Tensor, u_hat: tf.Tensor, **kwargs):
        temp = tf.reduce_mean(x[:, :, -1, :self.config.dim], axis=-1, keepdims=True) # (B, M, 1)
        float_min = tf.math.reduce_min(x[:, :, :, :self.config.dim], axis=2, keepdims=True) # (B, M, 1, d)
        temp_min = tf.reduce_mean(float_min, axis=-1) # (B,M,1)
        return temp - temp_min

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        """
        In this x has the dimension:
        (B, M, T, d+d) since this ia a concat with markovian variable
        """
        dim = self.config.dim
        T = self.config.T
        r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
        X_t = x[...,:dim]
        m_t = x[...,dim:]
        a_1 = (tf.math.log(X_t/m_t) + (r + vol ** 2/2) * (T - t))/(vol * tf.math.sqrt(T - t))
        a_2 = a_1 - vol * tf.math.sqrt(T - t)
        a_3 = a_1 - 2 * r/vol * tf.math.sqrt(T - t)
        y_t = X_t * dist.cdf(a_1) - m_t * tf.exp(-r * (T-t)) * dist.cdf(a_2) - \
            X_t * vol ** 2/(2 * r) * (dist.cdf(-a_1) - tf.exp(-r * (T-t)) * (m_t/X_t) ** (2 * r / vol**2) * dist.cdf(-a_3)) 
        return y_t

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
   
    def payoff(self, x_arg: tf.Tensor, param: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        x: tf.Tensor
            Asset price path. Tensor of shape (B, M, N, d + d)
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
        x = x_arg[...,:self.config.dim]
        geo_mean = tf.reduce_mean(tf.exp(tf.reduce_sum(tf.math.log(x), axis=2) * dt / T), axis=-1, keepdims=True)
        return tf.nn.relu(geo_mean - k)

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        """
        In this x has the dimension:
        (B, M, T, d+d) since this ia a concat with markovian variable
        """
        T = self.config.T
        r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
        K = tf.expand_dims(u_hat[:, :, :, 2], -1)
        y_t = x[...,:self.config.dim]
        G_t = x[...,self.config.dim:]
        mu_bar = (r - vol**2/2) * (T - t)**2/2/T
        vol_bar = vol/T * tf.math.sqrt((T - t) ** 3/3)
        d_2 = (t/T * tf.math.log(G_t) + (1 - t/T) * tf.math.log(y_t) + mu_bar - tf.math.log(K))/vol_bar
        d_1 = d_2 + vol_bar
        A = tf.exp(-r * (T - t))*(G_t ** (t/T) * y_t ** (1 - t/T) * tf.exp(mu_bar + vol_bar**2/2) * dist.cdf(d_1) - K * dist.cdf(d_2))
        return A


        

    




import tensorflow as tf
from typing import List, Tuple
import tensorflow_probability as tfp

tfd = tfp.distributions
dist = tfd.Normal(loc=0., scale=1.)
from scipy.stats import norm

N = norm.cdf


class DeepONet(tf.keras.Model):
    """
    The deep O net, The arguments are hidden layers of brunch and trunk net
    brunch_layer: The list of hidden sizes of trunk nets;
    trunk_layer: The list of hidden sizes of trunk nets
    """

    def __init__(self, branch_layer: List[int], trunk_layer: List[int]):
        super(DeepONet, self).__init__()
        self.branch = DenseNet(branch_layer)
        self.trunk = DenseNet(trunk_layer)

    def call(self, inputs: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        """
        The input of state can be either 3-dim or 4-dim but once fixed a problem the
        dimension of the input tensor is fixed.
        """
        time_tensor, state_tensor, parmeter_tensor = inputs
        br = self.branch(parmeter_tensor)
        tr = self.trunk(tf.concat([time_tensor, state_tensor], -1))
        value = tf.math.reduce_sum(br * tr, axis=-1, keepdims=True)
        return value

    def grad(self, inputs: Tuple[tf.Tensor]) -> tf.Tensor:
        _, state_tensor, _ = inputs
        with tf.GradientTape(watch_accessed_variables=True) as t:
            t.watch(state_tensor)
            out = self.call(inputs, training=False)
        grad = t.gradient(out, state_tensor)
        del t
        return grad


class DeepONetPath(tf.keras.Model):
    """
    The deep O net, The arguments are hidden layers of brunch and trunk net
    brunch_layer: The list of hidden sizes of trunk nets;
    trunk_layer: The list of hidden sizes of trunk nets
    """

    def __init__(self, branch_layer: List[int], trunk_layer: List[int]):
        super(DeepONetPath, self).__init__()
        self.branch = DenseNet(branch_layer)
        self.trunk = DenseNet(trunk_layer)

    def call(self, inputs: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        """
        The input of state can be either 3-dim or 4-dim but once fixed a problem the
        dimension of the input tensor is fixed.
        """
        time_tensor, state_tensor, path_embedded_tensor, parmeter_tensor = inputs
        br = self.branch(parmeter_tensor)
        tr = self.trunk(tf.concat([time_tensor, state_tensor, path_embedded_tensor], -1))
        value = tf.math.reduce_sum(br * tr, axis=-1, keepdims=True)
        return value

    def grad(self, inputs: Tuple[tf.Tensor]) -> tf.Tensor:
        _, state_tensor, _, _ = inputs
        with tf.GradientTape(watch_accessed_variables=True) as t:
            t.watch(state_tensor)
            out = self.call(inputs, training=False)
        grad = t.gradient(out, state_tensor)
        del t
        return grad


class DenseNet(tf.keras.Model):
    """
    The feed forward neural network for brunch and trunk net
    """

    def __init__(self, num_layers: List[int]):
        super(DenseNet, self).__init__()
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_layers))]

        self.dense_layers = [tf.keras.layers.Dense(num_layers[i],
                                                   kernel_initializer=tf.initializers.GlorotUniform(),
                                                   bias_initializer=tf.random_uniform_initializer(0.01, 0.05),
                                                   use_bias=True,
                                                   activation=None, )
                             for i in range(len(num_layers))]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense """
        for i in range(len(self.dense_layers)):
            x = self.bn_layers[i](x)
            x = self.dense_layers[i](x)
            x = tf.nn.relu(x)
        return x

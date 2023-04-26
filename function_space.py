import tensorflow as tf
from typing import List, Tuple
import tensorflow_probability as tfp

tfd = tfp.distributions
dist = tfd.Normal(loc=0., scale=1.)
from scipy.stats import norm
from typing import Optional

N = norm.cdf
class DenseNet(tf.keras.Model):
    """
    The feed forward neural network
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
        time_tensor, state_tensor, u_tensor = inputs
        br = self.branch(u_tensor)
        tr = self.trunk(tf.concat([time_tensor, state_tensor], -1))
        value = tf.math.reduce_sum(br * tr, axis=-1, keepdims=True)
        return value


class PermutationInvariantLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    '''
    permutation invariant layer for 2-D vector, it is only invariant for the dimension 
    of the stock but for dimension of each asset it is just a FNN. Then the function
    \varphi(x) is approximated by the conv1d layers.
    input_dim[-2] is the dim of stocks, input_dim[-1] is the dim of info in each stock
    eg: [128, 5, 2] means there are 5 stocks with each stocks having price and volatility data
    '''
    super(PermutationInvariantLayer, self).__init__()
    self.num_outputs = num_outputs
  
  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    
    self.bias = self.add_weight("bias",
                                  shape=[self.num_outputs,
                                        ])
    
  def call(self, inputs):
    output = tf.tensordot(inputs, self.kernel, [[-1], [0]]) + self.bias
    output = tf.nn.relu(output)
    return output
  
class DeepONetwithPI(DeepONet):
    def __init__(self, branch_layer: List[int], trunk_layer: List[int], pi_layer: List[int], num_assets: int=5):
        self.num_assets = num_assets
        super(DeepONetwithPI, self).__init__(branch_layer, trunk_layer)
        self.PI_layers = tf.keras.Sequential(layers=[PermutationInvariantLayer(m) for m in pi_layer]) #naming issue?
    
    def reshape_state(self, state: tf.Tensor):
        state_shape = tf.shape(state)
        dim = state_shape[-1]
        num_markov = int(dim/self.num_assets)
        return tf.reshape(state, [state_shape[0], \
                        state_shape[1], state_shape[2], self.num_assets, num_markov])

    def call(self, inputs: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        """
        state tensor is a multiple thing if each asset associated with more than 1 variable
        For example 
        under SV model we have state {(S_1, v_1), ..., (S_d, v_d)}, 
        then the dimension of the state is (B, M, N, d, 2)
        under SV for Asian option state is {(S_1, v_1, I_1), ..., (S_d, v_d, I_d)}, 
        then the dimension of the state is (B, M, N, d, 3)
        first we need to make (B, M, N, d * 3) to (B, M, N, d, 3)
        """
        time_tensor, state_tensor, u_tensor = inputs
        state_tensor = self.reshape_state(state_tensor)
        state_before_pi = self.PI_layers(state_tensor)
        state_after_pi  = tf.reduce_mean(state_before_pi, axis=-2)
        inputs_for_deeponet = time_tensor, state_after_pi, u_tensor
        return super(DeepONetwithPI, self).call(inputs_for_deeponet)

class DenseOperator(tf.keras.Model):
    """
    The 1D convolutional neural network with input of the function itself
    Input: the function supported on the time domain with shape: batch_shape + (time_steps, num_of_functions)
    Output: The flattened latent layer with shape: batch_shape + (num_outputs)
    """
    def __init__(self, num_outputs):
        super(DenseOperator, self).__init__()
        self.num_outputs = num_outputs
        self.w = tf.keras.layers.Dense(self.num_outputs, activation='relu')
    
    def call(self, x: tf.Tensor):
        # the x has shape batch_size + (time_steps, num_funcs), where batch_size is a 3-tuple
        # return: batch_size + (num_outputs)
        flat_dim = tf.shape(x)[-2] * tf.shape(x)[-1]
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], flat_dim])
        x = self.w(x)
        return x

class KernelOperator(DenseOperator):
    """
    The 1D convolutional neural network with input of the function itself
    Input: the function supported on the time domain with shape: batch_shape + (time_steps, num_of_functions)
    Output: The flattened latent layer with shape: batch_shape + (num_outputs)
    """
    def __init__(self, filters, strides, num_outputs):
        super(KernelOperator, self).__init__(num_outputs)
        self.filters = filters
        self.strides = strides
        self.conv1d = tf.keras.layers.Conv1D(self.filters, self.strides, padding='valid', activation='relu')
    
    def call(self, x: tf.Tensor):
        # the x has shape batch_size + (time_steps, num_funcs), where batch_size is a 3-tuple
        # return: batch_size + (num_outputs)
        x = self.conv1d(x)
        return super(KernelOperator, self).call(x)

class DeepKernelONetwithPI(DeepONetwithPI):
    def __init__(self, branch_layer: List[int], 
                 trunk_layer: List[int], 
                 pi_layer: List[int], 
                 num_assets: int=5, 
                 dense: bool=False,
                 num_outputs: int=6,
                 filters: Optional[int]=None, 
                 strides: Optional[int]=None):
        super(DeepKernelONetwithPI, self).__init__(branch_layer, trunk_layer, pi_layer, num_assets)
        if dense:
            self.kernelop = DenseOperator(num_outputs)
        else:
            self.kernelop = KernelOperator(filters, strides, num_outputs)
    
    def call(self, inputs: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        """
        we first let the function pass the kernel operator and then we flatten the hidden state
        and concat it with the input parameters and then we combine all of them into the brunch net
        for trunk net, things are all inherented from the deepOnet with PI. 
        The input is a tuple with 4 tensors (time, state, u_function, u_parmaters)
        Each has the dimension:
        t: batch_shape + (1)
        state: batch_shape + (dim_markov)
        u_function: batch_shape + (time_steps, num_functions)
        u_parameters: batch_shape + (num_parameters)
        """
        time_tensor, state_tensor, u_func, u_par = inputs
        latent_state = self.kernelop(u_func)
        u_tensor = tf.concat([latent_state, u_par], axis=-1)
        inputs_for_deeponetpi = time_tensor, state_tensor, u_tensor
        return super(DeepKernelONetwithPI, self).call(inputs_for_deeponetpi)




if __name__ == "__main__":
    import numpy as np
    inputs = tf.keras.layers.Input(shape=(None, None, 3,4))
    pi_layer_1= PermutationInvariantLayer(5)
    pi_layer_2= PermutationInvariantLayer(3)
    pp = pi_layer_2(pi_layer_1(inputs))
    x = np.array([[[[[1.,2.,3.,4.], [4.,3.,2.,1.], [3.,5.,4.,6.]], [[5.,4.,3.,2.], [2.,3.,4.,5.], [3.,5.,4.,6.]]]]])
    x_pi = np.array([[[[[4.,3.,2.,1.], [3.,5.,4.,6.], [1.,2.,3.,4.]], [[2.,3.,4.,5.], [3.,5.,4.,6.], [5.,4.,3.,2.]]]]])
    x = tf.tile(x, [2, 1, 1, 1, 1])
    x = tf.tile(x, [1, 2, 1, 1, 1])
    y = pi_layer_2(pi_layer_1(x))
    y_pi = pi_layer_2(pi_layer_1(x_pi))
    print(y)
    print(y_pi)
    model = tf.keras.Model(inputs=inputs, outputs=pp)
    print(model(x))
    print(model(x_pi))
    deeponet = DeepONetwithPI([3,3], [3,3], [6, 6], 10)
    assets = tf.random.normal([1, 1, 1, 30])
    t = tf.random.uniform([1, 1, 1, 1])
    u_hat = tf.random.normal([1, 1, 1, 4])
    y = deeponet((t, assets, u_hat))
    print(y)
    ko = KernelOperator(2, 10, 6)
    x = tf.random.normal((128, 30, 100, 50, 2))
    print(ko(x).shape)
    print(ko.summary())



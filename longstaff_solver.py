import tensorflow as tf
import numpy as np
from data_generators import DiffusionModelGenerator
from sde import ItoProcessDriver
from options import BaseOption
from function_space import DenseNet, DeepONet, DenseOperator, KernelOperator
from typing import List, Tuple, Optional
import sde as eqn
import options as opts
import time
DELTA_CLIP = 50

class SubONet(tf.keras.Model):
    def __init__(self, branch_layer: List[int], trunk_layer: List[int]):
        super(SubONet, self).__init__()
        self.branch = DenseNet(branch_layer)
        self.trunk = DenseNet(trunk_layer)
        self.bias = tf.Variable(tf.random.normal([1]))

    def call(self, inputs: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        """
        The input of state can be either 3-dim or 4-dim but once fixed a problem the
        dimension of the input tensor is fixed.
        state_tensor: [batch, samples, dim]
        u_tensor: [batch, samples, dim_of_functions]
        """
        state_tensor, u_tensor = inputs
        br = self.branch(u_tensor)
        tr = self.trunk(state_tensor)
        value = tf.math.reduce_sum(br * tr, axis=-1, keepdims=True) 
        b = tf.ones_like(value) * self.bias
        return value + b

class DenseSubOperator(tf.keras.Model):
    """
    The 1D convolutional neural network with input of the function itself
    Input: the function supported on the time domain with shape: batch_shape + (time_steps, num_of_functions)
    Output: The flattened latent layer with shape: batch_shape + (num_outputs)
    """
    def __init__(self, num_outputs):
        super(DenseSubOperator, self).__init__()
        self.num_outputs = num_outputs
        self.w = tf.keras.layers.Dense(self.num_outputs, activation='relu')
    
    def call(self, x: tf.Tensor):
        # the x has shape batch_size + (sensors, num_funcs), where batch_size is a 3-tuple
        # return: batch_size + (num_outputs)
        # batch_size = (B, M)
        flat_dim = tf.shape(x)[-2] * tf.shape(x)[-1]
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], flat_dim])
        x = self.w(x)
        return x
    
class KernelSubONet(SubONet):
    def __init__(self, branch_layer: List[int], 
                 trunk_layer: List[int],  
                 num_outputs: int=10):
        super(KernelSubONet, self).__init__(branch_layer, trunk_layer)
        self.kernelop = DenseSubOperator(num_outputs)
       
    def call(self, inputs: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        """
        we first let the function pass the kernel operator and then we flatten the hidden state
        and concat it with the input parameters and then we combine all of them into the brunch net
        for trunk net, things are all inherented from the deepOnet. 
        The input is a tuple with 3 tensors (state, u_function, u_parmaters)
        Each has the dimension:
        t: batch_shape + (1)
        state: batch_shape + (dim_markov)
        u_function: batch_shape + (sensors, num_functions)
        u_parameters: batch_shape + (num_parameters)
        batch_shape = (B, M)
        """
        state_tensor, u_func, u_par = inputs
        latent_state = self.kernelop(u_func)
        u_tensor = tf.concat([latent_state, u_par], axis=-1) # (B, M , d1+d2)
        inputs = state_tensor, u_tensor
        return super(KernelSubONet, self).call(inputs)



class LongStaffSolver:
    def __init__(self, sde: ItoProcessDriver, option: BaseOption, config):
        self.sde = sde
        self.option = option
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.exercise_date = self.eqn_config.exercise_date
        self.exercise_index = self.option.exer_index #[40, 60] <=> [0, 1] len=2 (for index) is the time index for early exercise
        self.models = [SubONet(self.net_config.branch_layers, self.net_config.trunk_layers)   
                        for _ in range (len(self.exercise_index))]
        self.num_batches = 100
        self.data_generator = DiffusionModelGenerator(self.sde, self.eqn_config, self.option, self.num_batches)
        
    def loss_fn(self, x: tf.Tensor, u: tf.Tensor, idx: int, training=None):
        """
        calculate the batch loss of x and u
        x: [B, M, N, d], but we will get a slice then it is [B, M, d]
        idx: it denotes the index of the sub network
        """
        T = self.eqn_config.T
        r = tf.expand_dims(u[:, :, 0, 0], -1)
        if idx == len(self.exercise_index)-1:
            discount_factor = tf.exp(-r * (T - self.exercise_date[idx]))
            payoff = self.g_tf(x, u)
            target = discount_factor * payoff
        
        else:
            discount_factor = tf.exp(-r * (self.exercise_date[idx+1] - self.exercise_date[idx]))
            index_future = self.exercise_index[idx+1]
            early_exercise_payoff = self.g_tf(x[:, :, :index_future, :], u[:, :, :index_future, :])
            continuation_value = self.models[idx+1]((x[:, :, index_future, :], u[:, :, index_future, :]), training=False)
            payoff = tf.maximum(continuation_value, early_exercise_payoff)
            target = tf.stop_gradient(discount_factor * payoff)
        index_now = self.exercise_index[idx]
        value_now = self.models[idx]((x[:, :, index_now, :], u[:, :, index_now, :]), training)
        delta = value_now - target
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                    2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        dv = tf.reduce_mean(tf.abs(delta))
        return loss, dv

    @tf.function 
    def train_step(self, x: tf.Tensor, u: tf.Tensor, idx: int):
        
        with tf.GradientTape() as tape:
            loss, dv = self.loss_fn(x, u, idx, training=True)
            grad = tape.gradient(loss, self.models[idx].trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.models[idx].trainable_variables))  
        return loss, dv
    
    def train(self):
        for idx in reversed(range(len(self.exercise_index))):
            learning_rate = self.net_config.lr
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=200,
            decay_rate=0.9)    
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
            start_time = time.time()
            for epoch in range(self.net_config.epochs):
                for (i, data) in enumerate(self.data_generator):
                    _, x, _, u = data[0]
                    loss, dv = self.train_step(x, u, idx)
                    eclapsed_time = time.time() - start_time
                    if i%10==0:
                        print(f"In {epoch+1}-th epoch the MAE of the net {idx+1} is {dv}, loss is {loss}, time eclapse {eclapsed_time}\n")


    def g_tf(self, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        payoff = self.option.payoff(x, u_hat)
        return payoff


if __name__ == '__main__':
    import json
    import munch

    sde_list = ["GBM", "TGBM", "SV", "CEV", "SVJ"]
    option_list = ["European", "EuropeanPut", "Lookback", "Asian", "Basket", "BasketnoPI", "Swap", "TimeEuropean", "BermudanPut"]
    dim_list = [1, 3, 5, 10, 20]
    def main(sde_name: str, option_name: str, dim: int=1):
        if (sde_name not in sde_list) or (option_name not in option_list) or (dim not in dim_list):
            raise ValueError(f"please input right sde_name in {sde_list},\
                            option_name in {option_list} and dim in {dim_list}")
        else:
            json_path = f'./config/{sde_name}_{option_name}_{dim}.json'
        with open(json_path) as json_data_file:
            config = json.load(json_data_file)
        
        config = munch.munchify(config)
        sde = getattr(eqn, config.eqn_config.sde_name)(config)
        option = getattr(opts, config.eqn_config.option_name)(config)
        solver = LongStaffSolver(sde, option, config)
        solver.train()
    
    main("GBM", "BermudanPut", 3)
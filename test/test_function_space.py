import tensorflow as tf
from function_space import PermutationInvariantLayer, DeepONetwithPI
import numpy as np

class TestPermutationInvariantLayer(tf.test.TestCase):
    def testPIlayer(self):
        pi_layer_1= PermutationInvariantLayer(5)
        pi_layer_2= PermutationInvariantLayer(3)
        x    = np.array([[[[[1.,2.,3.,4.], [4.,3.,2.,1.], [3.,5.,4.,6.]], 
                           [[5.,4.,3.,2.], [2.,3.,4.,5.], [3.,5.,4.,6.]]]]])
        x_pi = np.array([[[[[4.,3.,2.,1.], [3.,5.,4.,6.], [1.,2.,3.,4.]], 
                           [[2.,3.,4.,5.], [3.,5.,4.,6.], [5.,4.,3.,2.]]]]])
        y = pi_layer_2(pi_layer_1(x))
        y_pi = pi_layer_2(pi_layer_1(x_pi))
        self.assertAllLessEqual(tf.abs(tf.reduce_sum(y, axis=-2) - tf.reduce_sum(y_pi, axis=-2)), 1e-6)

class TestDeepONetwithPI(tf.test.TestCase):
    def testnumberofparams(self):
        l = 1
        m = 6
        d = 3
        N = 10
        B = 2
        M = 2
        T = 2
        deeponet = DeepONetwithPI([3,3], [3,3], [m] * (l + 1), 10)
        assets = tf.random.normal([B, M, T, N * d])
        t = tf.random.uniform([B, M, T, 1])
        u_hat = tf.random.normal([B, M, T, 4])
        y = deeponet((t, assets, u_hat))
        def num_params_pi(m, d, l):
            return m * (d + 1) + m * (m + 1) * (l - 1) + (m + 1) * m
        num_actual = 0
        for v in deeponet.PI_layers.trainable_weights: 
          num_actual += tf.size(v)
        self.assertEqual(num_params_pi(m, d, l), num_actual)

    def testPIproperty(self):
        l = 1
        m = 6
        d = 3
        N = 10
        B = 2
        M = 2
        T = 2
        deeponet = DeepONetwithPI([3,3], [3,3], [m] * (l + 1), 10)
        assets = tf.random.normal([B, M, T, N * d])
        t = tf.random.uniform([B, M, T, 1])
        u_hat = tf.random.normal([B, M, T, 4])
        y_ni = deeponet((t, assets, u_hat))
        assets_pi = tf.reverse(deeponet.reshape_state(assets), [3])
        assets_pi = tf.reshape(assets_pi, [B, M, T, N * d])
        y_pi = deeponet((t, assets_pi, u_hat))
        self.assertAllLessEqual(tf.reduce_sum(tf.abs(y_ni - y_pi)), 1e-7)


        
        
        
if __name__ == "__main__":
    tf.test.main()


import tensorflow as tf
from options import EuropeanOption, GeometricAsian, LookbackOption
from sde import GeometricBrowianMotion
from config import Config

class TestEuropeanOption(tf.test.TestCase):
    def testPayoff(self):
        config = Config()
        opt = EuropeanOption(config)
        x = tf.random.uniform([3,5,6,2]) + 0.1
        u_hat = tf.random.uniform([3,5,6,4]) 
        payoff = opt.payoff(x, u_hat)
        self.assertEqual(payoff.shape, [3,5,1])

    def testExactprice(self):
        pass

    def testSampleparameters(self):
        pass


class TestGeoAsianOption(tf.test.TestCase):
    def testMarkovVar(self):
        config = Config()
        opt = GeometricAsian(config)

        x = tf.constant([[[0.2, 1, 3, 0.8],[0.7, 4, 0.8, 3]],
                         [[0.2, 1, 3, 0.8],[0.7, 4, 0.8, 3]],
                         [[0.2, 1, 3, 0.8],[0.7, 4, 0.8, 3]]])
        
        g = tf.constant([[[0.2, 0.4472,  0.84343, 0.83236],[0.7, 1.6733, 1.3084, 1.61006]],
                         [[0.2, 0.4472,  0.84343, 0.83236],[0.7, 1.6733, 1.3084, 1.61006]],
                         [[0.2, 0.4472,  0.84343, 0.83236],[0.7, 1.6733, 1.3084, 1.61006]]])

        g_hat = opt.markovian_var(x)
        x_test = tf.random.uniform([3,5,6,2]) + 0.1
        g_test = opt.markovian_var(x_test)
        dt = config.dt
        for i in range(5):
            geo_avg = tf.exp(tf.reduce_sum(tf.math.log(x_test[:, :, :i+1, :]), axis=2) * dt/((i+1) * dt))
            self.assertAllLessEqual(tf.reduce_mean(tf.math.abs(g_test[:, :, i, :] - geo_avg)), 1e-6)
        self.assertAllEqual(g_test.shape, x_test.shape, msg="shape is not same")
        self.assertAllLessEqual(tf.reduce_mean(tf.math.abs(g_hat - g)), 1e-4)
        

    def testPayoff(self):
        config = Config()
        opt = GeometricAsian(config)
        x = tf.random.uniform([3,5,6,1]) + 0.1
        u_hat = tf.random.uniform([3,5,6,4]) 
        payoff = opt.payoff(x, u_hat)
        self.assertEqual(payoff.shape, [3,5,1])
        

    def testExactprice(self):
        pass


class TestlookbackOption(tf.test.TestCase):
    def testMarkovVar(self):
        config = Config()
        opt = LookbackOption(config)
        x = tf.constant([[[2,1,3,0,-1],[7,4,8,3,2]], 
                        [[4,5,3,4,2], [6,5,23,8,1]], 
                        [[2,4,1,-7,-78],[6,5,7,8,9]]])
        
        m = tf.constant([[[2,1,1,0,-1],[7,4,4,3,2]], 
                        [[4,4,3,3,2], [6,5,5,5,1]], 
                        [[2,2,1,-7,-78],[6,5,5,5,5]]])
        
        m_hat = opt.markovian_var(tf.expand_dims(x, -1))

        x_test = tf.random.uniform([3,5,6,2])
        m_test = opt.markovian_var(x_test)
        for i in range(5):
            self.assertAllEqual(m_test[:, :, i, :], tf.reduce_min(x_test[:, :, :i+1, :], axis=2), msg="wrong 1")
        self.assertAllEqual(m_test[:, :, -1, :], tf.reduce_min(x_test, axis=2), msg="wrong 2")
        self.assertAllEqual(m_hat, tf.expand_dims(m, -1), msg="wrong 3")
        self.assertAllEqual(m_test.shape, x_test.shape, msg="wrong 4")

    
    def testPayoff(self):
        config = Config()
        opt = LookbackOption(config)
        x = tf.random.uniform([3,5,6,1]) + 0.1
        u_hat = tf.random.uniform([3,5,6,4]) 
        payoff = opt.payoff(x, u_hat)
        self.assertEqual(payoff.shape, [3,5,1])
        x_min = tf.reduce_min(x, axis=2)
        self.assertAllEqual(payoff, x[:,:,-1,:] - x_min)

    def testExactprice(self):
        config = Config()
        opt = LookbackOption(config)
        config = Config(dim=1, iid=True)
        sde = GeometricBrowianMotion(config)
        u_hat = sde.sample_parameters(1)
        dim = config.dim
        x,_ = sde.sde_simulation(u_hat, config.M)
        x_m = opt.markovian_var(x)
        arg_x = tf.concat([x, x_m], axis=-1)
        time_stamp = tf.range(0, config.T, config.dt)
        time_steps = int(config.T / config.dt)
        time_stamp = tf.reshape(time_stamp, [1, 1, time_steps, 1])
        t = tf.tile(time_stamp, [u_hat.shape[0], config.M, 1, 1])
        u_hat = sde.expand_batch_inputs_dim(u_hat)
        price = opt.exact_price(t, arg_x, u_hat)
        exact_terminal_payoff = x[:,:,-1,:] - tf.reduce_min(x, 2)
        self.assertAllLessEqual(tf.abs(price[:, :, -1, :] - exact_terminal_payoff), 0.05)
        


if __name__ == "__main__":
    tf.test.main()
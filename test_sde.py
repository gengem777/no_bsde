import tensorflow as tf
import numpy as np
from sde import CEVModel, HestonModel, GeometricBrowianMotion
from config import Config
class TestGeometricBrowianMotion(tf.test.TestCase):
    def test_martingale_property(self):
        config = Config(M=10000, dim=10, independent=False)
        sde = GeometricBrowianMotion(config)
        u_hat = tf.constant([[0.05, 0.1, 0.2],[0.05, 0.2, 1/(1-config.dim)+1e-4]])
        dim = config.dim
        s,_ = sde.sde_simulation(u_hat, config.M)
        s_exp = tf.reduce_mean(s[:, :, -1, :dim], axis=1) * tf.exp(-0.05)
        s_init = tf.reduce_mean(s[:, :, 0, :dim], axis=1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(s_init - s_exp)), 1e-2)

class TestHestonModel(tf.test.TestCase):
    def test_initial_sampler(self):
        config = Config()
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04],[0.04, 0.58, 1.06, 1.06]])
        dim = config.dim
        samples = config.M
        initial_state = sde.initial_sampler(u_hat)
        self.assertEqual(initial_state.shape, [2, samples, 2*dim])

    def test_drift(self):
        config = Config()
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04],[0.04, 0.58, 1.06, 1.06]])
        state = tf.constant([[1.2, 0.12],[1.13, 0.11]])
        a_1 = 0.05 * 1.2
        b_1 = 0.04 * 1.13
        a_2 = 0.45 * (1.05 - 0.12)
        b_2 = 0.58 * (1.06 - 0.11)
        drift = tf.constant([[a_1, a_2], [b_1, b_2]])
        dim = config.dim
        M = config.M

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, dim*2])
            tensor3d = tf.tile(tensor, [1, M, 1])
            return tensor3d
        
        state = expand_dim(state)
        drift = expand_dim(drift)
        drift_to_test = sde.drift(0, state, u_hat)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(drift - drift_to_test)), 1e-6)

    def test_diffusion(self):
        config = Config()
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04],[0.04, 0.58, 1.06, 1.06]])
        state = tf.constant([[1.2, 0.12],[1.13, 0.11]])
        v_1 = np.sqrt(0.12) * 1.04
        v_2 = np.sqrt(0.11) * 1.06
        s_1 = np.sqrt(0.12) * 1.2
        s_2 = np.sqrt(0.11) * 1.13
        diff = tf.constant([[s_1, v_1], [s_2, v_2]])

        dim = config.dim
        M = config.M

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, dim*2])
            tensor3d = tf.tile(tensor, [1, M, 1])
            return tensor3d
        
        state = expand_dim(state)
        diff = tf.cast(expand_dim(diff), tf.float32)
        diff_to_test = sde.diffusion(0, state, u_hat)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(diff - diff_to_test)), 1e-6)

    def test_drift_onestep(self):
        config = Config()
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04], [0.05, 0.45, 1.05, 1.04]])
        state = tf.constant([[1.2, 0.12],[1.13, 0.11]])
        dim = config.dim
        M = config.M
        N = config.time_steps

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, 1, dim*2])
            tensor4d = tf.tile(tensor, [1, M, N, 1])
            return tensor4d
        
        state_tensor = expand_dim(state)
        u_hat = tf.reshape(u_hat, [u_hat.shape[0], 1, 1, u_hat.shape[-1]])
        u_hat = tf.tile(u_hat, [1, M, N, 1])
        drift_state = sde.drift_onestep(0, state_tensor, u_hat)
        s = state_tensor[..., :dim]
        v = state_tensor[..., dim:]
        s_plus = s * 0.05
        v_plus = 0.45 * (1.05 - v)
        drift_target = tf.concat([s_plus, v_plus], axis=-1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(drift_state - drift_target)), 1e-6)

    def test_diffusion_onestep(self):
        config = Config()
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04], [0.05, 0.45, 1.05, 1.04]])
        state = tf.constant([[1.2, 0.12],[1.13, 0.11]])
        dim = config.dim
        M = config.M
        N = config.time_steps

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, 1, dim*2])
            tensor4d = tf.tile(tensor, [1, M, N, 1])
            return tensor4d
        
        state_tensor = expand_dim(state)
        u_hat = tf.reshape(u_hat, [u_hat.shape[0], 1, 1, u_hat.shape[-1]])
        u_hat = tf.tile(u_hat, [1, M, N, 1])
        diff_state = sde.diffusion_onestep(0, state_tensor, u_hat)
        s = state_tensor[..., :dim]
        v = state_tensor[..., dim:]
        s_plus = tf.sqrt(v) * s
        v_plus = tf.sqrt(v) * 1.04
        diff_target = tf.concat([s_plus, v_plus], axis=-1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(diff_state - diff_target)), 1e-6)

    def test_martingale_property(self):
        """
        this is integrate test to test whether under Q measure the discounted process is a martingale
        """
        config = Config(M=10000)
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04],[0.05, 0.58, 1.06, 1.06]])
        dim = config.dim
        s,_ = sde.sde_simulation(u_hat, config.M)
        s_exp = tf.reduce_mean(s[:, :, -1, :dim], axis=1) * tf.exp(-0.05)
        s_init = tf.reduce_mean(s[:, :, 0, :dim], axis=1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(s_init - s_exp)), 1e-2)

class TestCEVModel(tf.test.TestCase):
    def test_initial_sampler(self):
        config = Config()
        sde = CEVModel(config)
        u_hat = tf.constant([[0.05, 0.04, -0.05],[0.04, 0.05, 0.06]])
        dim = config.dim
        samples = config.M
        initial_state = sde.initial_sampler(u_hat)
        self.assertEqual(initial_state.shape, [2, samples, dim])
    
    def test_drift(self):
        config = Config()
        sde = CEVModel(config)
        u_hat = tf.constant([[0.05, 0.04, -0.05],[0.04, 0.05, 0.06]])
        state = tf.constant([[1.2],[1.13]])
        drift = tf.constant([[0.05 * 1.2], [0.04 * 1.13]])
        dim = config.dim
        M = config.M

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, dim])
            tensor3d = tf.tile(tensor, [1, M, 1])
            return tensor3d
        
        state = expand_dim(state)
        drift = expand_dim(drift)
        drift_to_test = sde.drift(0, state, u_hat)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(drift - drift_to_test)), 1e-6)

    def test_diffusion(self):
        config = Config()
        sde = CEVModel(config)
        u_hat = tf.constant([[0.05, 0.04, -0.05],[0.04, 0.05, 0.06]])
        state = tf.constant([[1.2],[1.13]])
        s_1 = 0.04 * 1.2 ** (-0.05)
        s_2 = 0.05 * 1.13 ** 0.06
        diff = tf.constant([[s_1], [s_2]])

        dim = config.dim
        M = config.M

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, dim])
            tensor3d = tf.tile(tensor, [1, M, 1])
            return tensor3d
        
        state = expand_dim(state)
        diff = tf.cast(expand_dim(diff), tf.float32)
        diff_to_test = sde.diffusion(0, state, u_hat)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(diff - diff_to_test)), 1e-6)

    def test_drift_onestep(self):
        config = Config()
        sde = CEVModel(config)
        u_hat = tf.constant([[0.05, 0.04, -0.05],[0.05, 0.05, 0.06]])
        state = tf.constant([[1.2],[1.13]])
        dim = config.dim
        M = config.M
        N = config.time_steps

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, 1, dim])
            tensor4d = tf.tile(tensor, [1, M, N, 1])
            return tensor4d
        
        state_tensor = expand_dim(state)
        u_hat = tf.reshape(u_hat, [u_hat.shape[0], 1, 1, u_hat.shape[-1]])
        u_hat = tf.tile(u_hat, [1, M, N, 1])
        drift_state = sde.drift_onestep(0, state_tensor, u_hat)
        s = state_tensor[..., :dim]
        s_plus = s * 0.05
        drift_target = s_plus
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(drift_state - drift_target)), 1e-6)
    
    def test_diffusion_onestep(self):
        config = Config()
        sde = CEVModel(config)
        u_hat = tf.constant([[0.05, 0.04, 0.06],[0.05, 0.04, 0.06]])
        state = tf.constant([[1.2],[1.13]])
        dim = config.dim
        M = config.M
        N = config.time_steps

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, 1, dim])
            tensor4d = tf.tile(tensor, [1, M, N, 1])
            return tensor4d
        
        state_tensor = expand_dim(state)
        u_hat = tf.reshape(u_hat, [u_hat.shape[0], 1, 1, u_hat.shape[-1]])
        u_hat = tf.tile(u_hat, [1, M, N, 1])
        diff_state = sde.diffusion_onestep(0, state_tensor, u_hat)
        s = state_tensor[..., :dim]
        s_plus =   0.04 * s ** 0.06
        diff_target = s_plus
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(diff_state - diff_target)), 1e-6)

    def test_martingale_property(self):
        """
        this is integrate test to test whether under Q measure the discounted process is a martingale
        """
        config = Config(M=10000)
        sde = CEVModel(config)
        u_hat = tf.constant([[0.03, 0.04, -0.05],[0.03, 0.05, 0.06]])
        dim = config.dim
        s,_ = sde.sde_simulation(u_hat, config.M)
        s_exp = tf.reduce_mean(s[:, :, -1, :dim], axis=1) * tf.exp(-0.03 * config.T)
        s_init = tf.reduce_mean(s[:, :, 0, :dim], axis=1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(s_init - s_exp)), 1e-2)
    

if __name__ == "__main__":
    tf.test.main()

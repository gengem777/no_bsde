import tensorflow as tf
from data_generators import DiffusionModelGenerator
from markov_solver import MarkovianSolver, BaseBSDESolver, BSDEMarkovianModel
from sde import GeometricBrowianMotion, CEVModel, HestonModel
from options import EuropeanOption, GeometricAsian, LookbackOption
from config import Config

class TestMarkovianSolver(tf.test.TestCase):
    def test_gbm_european(self):
        config = Config()
        sde = GeometricBrowianMotion(config)
        option = EuropeanOption(config)
        markov_model = BSDEMarkovianModel(sde, option, config)
        # train phase
        markov_model.train(10)
        # validate phase
        markov_model.pre_setting()
        data_generator = DiffusionModelGenerator(sde, config, option, 20)
        inputs = data_generator.__getitem__(1)
        with tf.GradientTape() as tape:
            _, loss, _, _ = markov_model.model(inputs[0])
                # tf.print(loss, loss_int, loss_tml)
            l = tf.reduce_mean(loss)
            grad = tape.gradient(l, markov_model.model.trainable_variables)
        norm = 0.0
        N = 0
        for g in grad:
            norm = norm + tf.reduce_sum(g**2)
            N = N + tf.reduce_sum(g**2) / tf.reduce_mean(g**2)
        norm = tf.math.sqrt(norm)/N
        self.assertAllLessEqual(norm, 1e-3)
    
    def test_cev_european(self):
        config = Config()
        sde = CEVModel(config)
        option = EuropeanOption(config)
        markov_model = BSDEMarkovianModel(sde, option, config)
        # train phase
        markov_model.train(10)
        # validate phase
        markov_model.pre_setting()
        data_generator = DiffusionModelGenerator(sde, config, option, 20)
        inputs = data_generator.__getitem__(1)
        with tf.GradientTape() as tape:
            _, loss, _, _ = markov_model.model(inputs[0])
                # tf.print(loss, loss_int, loss_tml)
            l = tf.reduce_mean(loss)
            grad = tape.gradient(l, markov_model.model.trainable_variables)
        norm = 0.0
        N = 0
        for g in grad:
            norm = norm + tf.reduce_sum(g**2)
            N = N + tf.reduce_sum(g**2) / tf.reduce_mean(g**2)
        norm = tf.math.sqrt(norm)/N
        self.assertAllLessEqual(norm, 1e-3)
    
    def test_sv_european(self):
        config = Config()
        sde = HestonModel(config)
        option = EuropeanOption(config)
        markov_model = BSDEMarkovianModel(sde, option, config)
        # train phase
        markov_model.train(10)
        # validate phase
        markov_model.pre_setting()
        data_generator = DiffusionModelGenerator(sde, config, option, 20)
        inputs = data_generator.__getitem__(1)
        with tf.GradientTape() as tape:
            _, loss, _, _ = markov_model.model(inputs[0])
                # tf.print(loss, loss_int, loss_tml)
            l = tf.reduce_mean(loss)
            grad = tape.gradient(l, markov_model.model.trainable_variables)
        norm = 0.0
        N = 0
        for g in grad:
            norm = norm + tf.reduce_sum(g**2)
            N = N + tf.reduce_sum(g**2) / tf.reduce_mean(g**2)
        norm = tf.math.sqrt(norm)/N
        self.assertAllLessEqual(norm, 1e-3)



class TestMarkovianModel(tf.test.TestCase):
    pass

if __name__ == "__main__":
    tf.test.main()
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
        markov_model.train(5)
        # validate phase
        markov_model.pre_setting()
        data_generator = DiffusionModelGenerator(sde, config, option)
        inputs = data_generator.__getitem__(1)
        with tf.GradientTape() as tape:
            _, loss, _, _ = markov_model.model(inputs[0])
                # tf.print(loss, loss_int, loss_tml)
            loss = tf.reduce_mean(loss)
            grad = tape.gradient(loss, markov_model.model.trainable_variables)
        print(grad)
        print(loss)



class TestMarkovianModel(tf.test.TestCase):
    pass

if __name__ == "__main__":
    tf.test.main()
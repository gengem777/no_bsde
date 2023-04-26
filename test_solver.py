import tensorflow as tf
import numpy as np
from data_generators import DiffusionModelGenerator
from markov_solver import MarkovianSolver, BaseBSDESolver, BSDEMarkovianModel
from sde import GeometricBrowianMotion, CEVModel, HestonModel
from options import EuropeanOption, GeometricAsian, LookbackOption
from config import Config

class TestMarkovianSolver(tf.test.TestCase):
    def test_convergence(self):
        """
        In this problem, empirically, the loss come under 0.005 yields the expected 
        result, in this block we check whether the loss come below the thresold for
        some epochs
        """
        config = Config()
        sde = GeometricBrowianMotion(config)
        option = EuropeanOption(config)
        markov_model = BSDEMarkovianModel(sde, option, config)
        # train phase
        history = markov_model.train(10)
        loss = history.history['loss']
        averageloss = np.mean(loss[5:])
        self.assertAllLessEqual(averageloss, 5e-3)
    
    def test_cev_european(self):
        config = Config()
        sde = CEVModel(config)
        option = EuropeanOption(config)
        markov_model = BSDEMarkovianModel(sde, option, config)
        # train phase
        history = markov_model.train(10)
        loss = history.history['loss']
        averageloss = np.mean(loss[5:])
        self.assertAllLessEqual(averageloss, 5e-3)
        
    def test_sv_european(self):
        config = Config()
        sde = HestonModel(config)
        option = EuropeanOption(config)
        markov_model = BSDEMarkovianModel(sde, option, config)
        # train phase
        history = markov_model.train(10)
        loss = history.history['loss']
        averageloss = np.mean(loss[5:])
        self.assertAllLessEqual(averageloss, 5e-3)



class TestMarkovianModel(tf.test.TestCase):
    pass

if __name__ == "__main__":
    tf.test.main()
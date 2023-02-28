from sde import GeometricBrowianMotion
from options import EuropeanOption
from markov_solver import BSDEMarkovianModel
from config import Config

myconfig = Config()
mysde = GeometricBrowianMotion(myconfig)
myoption = EuropeanOption(myconfig)
mymodel = BSDEMarkovianModel(mysde, myoption, myconfig)
mymodel.train(10)
from sde import GeometricBrownianMotion, CEVModel, HestonModel
from options import EuropeanOption, GeometricAsian, LookbackOption, EuropeanBasketOption
from markov_solver import BSDEMarkovianModel
from config import Config
import numpy as np
import time

# myconfig = Config()
# gbm = GeometricBrowianMotion(myconfig)
# cev = CEVModel(myconfig)
# myoption = EuropeanOption(myconfig)
# mymodel = BSDEMarkovianModel(gbm, myoption, myconfig)
# print('begin GBM!')
# mymodel.train(1)
# mymodel = BSDEMarkovianModel(cev, myoption, myconfig)
# print('begin CEV!')
# mymodel.train(1)

def main(epoch: int, sde_name: str, option_name: str, multidim: bool=False, pi: bool=False):
    if not multidim:
        config = Config()
    else:
        if pi:
            config = Config(dim=10, iid=False)
            config.n_hidden = 20
        else:
            config = Config(dim=10, iid=True)
            config.n_hidden = 50

    if sde_name == 'GBM':
        sde = GeometricBrowianMotion(config)
    
    elif sde_name == 'CEV':
        sde = CEVModel(config)
    
    elif sde_name == 'SV':
        sde = HestonModel(config)
    else:
        raise ValueError("wrong SDE name")
    
    if option_name == 'European':
        option = EuropeanOption(config)

    elif option_name == 'Asian':
        option = GeometricAsian(config)

    elif option_name == 'Lookback':
        option = LookbackOption(config)
    
    elif option_name == 'Basket_PI' or option_name == 'Basket_wo_PI' or "Basket":
        option = EuropeanBasketOption(config)

    else:
        raise ValueError("wrong OPTION name")
    
    model = BSDEMarkovianModel(sde, option, config)
    print(config.n_hidden)
    print(f'begin train {option_name} under {sde_name} {config.dim} dimensions')
    checkpoint_path = f'checkpoint/{sde_name}_{option_name}_{config.dim}'
    time_start = time.time()
    history = model.train(epoch, checkpoint_path)  
    time_end = time.time()
    loss = history.losses
    print(f"time consume: {time_end - time_start}")
    np.savetxt(f'train_curve/{option_name}_under_{sde_name}_{config.dim}train.txt', loss)
    
    
        
if __name__ == "__main__":
    # main(10, "GBM", "European")
    # main(100, "GBM", "Asian")
    # main(10, "GBM", "Lookback")
    
    # main(10, "CEV", "European")
    # main(10, "GBM", "European", True)
    main(15, "GBM", "Basket_PI", True, pi=True)
    # main(10, "GBM", 'Basket_wo_PI', True, pi=False)
    # main(10, "SV", "European")


    # main("CEV", "Asian")
    # main("CEV", "Lookback")
    
    
    

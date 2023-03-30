from sde import GeometricBrowianMotion, CEVModel, HestonModel
from options import EuropeanOption, GeometricAsian, LookbackOption
from markov_solver import BSDEMarkovianModel
from config import Config
import numpy as np

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

def main(sde_name: str, option_name: str):
    config = Config()

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

    else:
        raise ValueError("wrong OPTION name")
    
    model = BSDEMarkovianModel(sde, option, config)
    print(f'begin train {option_name} under {sde_name}')
    history = model.train(20)  
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(f'train loss is: {loss} , val_loss is: {val_loss}')
    np.savetxt(f'no_bsde/output/{option_name}_under_{sde_name}_train.txt', loss)

        
if __name__ == "__main__":
    # main("GBM", "European")
    # main("CEV", "European")
    # main("GBM", "Asian")
    # main("GBM", "Lookback")
    # main("SV", "European")
    main("CEV", "Asian")
    main("CEV", "Lookback")
    
    

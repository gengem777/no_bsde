class Config:
    def __init__(self, T: float=1.0, dt: float=0.01, M: int=30, dim: int=1, batch_size: int=200, lr: float=0.005, 
    initial_mode: str='partial_fixed', x_init: float=1.0, vol_init: float=0.1, iid:bool=True, alpha: float=1.0, sensors: int=50):
        self.T = T  # time horizon
        self.dt = dt  # step size
        self.time_steps = int(self.T/self.dt)  # num of steps
        self.M = M  # path number of each input function
        self.dim = dim  # dimension of asset
        self.batch_size = batch_size  # batch size
        self.lr = lr  # learning rate 
        self.initial_mode = initial_mode  # 'fixed', 'partial_fixed', 'random'
        self.x_init = x_init # 'initial point of the whole problem'
        self.vol_init = vol_init  # 'initial vol of the whole problem'
        self.iid = iid
        self.n_hidden = 20
        self.n_layers = 4
        self.alpha = alpha
        self.sensors = sensors
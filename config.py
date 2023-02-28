class Config:
    def __init__(self, T: float=1.0, dt: float=0.01, samples: int=30, dim: int=1, batch_size: int=128, lr: float=0.01, 
    initial_mode: str='partial_fixed', x_init: float=1.0, is_Maliar: bool=False, alpha: float=100):
        self.T = T
        self.dt = dt
        self.time_steps = int(self.T/self.dt)
        self.M = samples
        self.dim = dim
        self.batch_size = batch_size
        self.lr = lr
        self.initial_mode = initial_mode
        self.x_init = x_init
        self.n_hidden = 10
        self.n_layers = 3
        self.n_a = 12
        self.is_Maliar = is_Maliar
        self.alpha = alpha
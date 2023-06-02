import tensorflow as tf
from base_solver import BaseBSDESolver
from data_generators import DiffusionModelGenerator
from longstaff_solver import LongStaffSolver
from options import BaseOption
from typing import List, Tuple
from sde import ItoProcessDriver

class BSDETrainer:
    def __init__(self, solver: BaseBSDESolver):
        self.solver = solver
        self.eqn_config = self.solver.eqn_config
        self.net_config = self.solver.net_config
        self.option = self.solver.option
        self.sde = self.solver.sde

    def pre_setting(self):
        learning_rate = self.net_config.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=200,
            decay_rate=0.9
        )
        Optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
        self.data_generator = DiffusionModelGenerator(self.sde, self.eqn_config, self.option, 100)
        # self.val_generator = DiffusionModelGenerator(self.sde, self.config, self.option, 20)
        self.solver.compile(optimizer=Optimizer)

    def train(self, nr_epochs: int, checkpoint_path: str):
        self.pre_setting()
        # Create a callback that saves the model's batch loss
        class LossHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []
            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
        history = LossHistory()
        self.solver.fit(x=self.data_generator, epochs=nr_epochs, callbacks=[history])
        self.solver.no_net.save_weights(checkpoint_path)
        # return history
        return history
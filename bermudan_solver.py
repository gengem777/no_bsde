import tensorflow as tf
from base_solver import BaseBSDESolver
from markov_solver import MarkovianSolver
from data_generators import DiffusionModelGenerator
from options import BaseOption
from typing import List, Tuple
from sde import ItoProcessDriver, HestonModel, TimeDependentGBM



class BermudanSolver(MarkovianSolver):
    """
    This is the class to construct the tain step of the BSDE loss function, the data is generated from three
    external classes: the sde class which yield the data of input parameters. the option class which yields the
    input function data and give the payoff function and exact option price.
    """

    def __init__(self, sde, option, config):
        super(BermudanSolver, self).__init__(sde, option, config)
        self.exer_index = self.option.exer_index

    def call(self, data: Tuple[tf.Tensor], training=None) -> Tuple[tf.Tensor]:
        """
        :param t: time B x M x (T-1) x 1
        :param x: state B x M x (T-1) x D
        :param  param: B x K ->(repeat) B x M x (T-1) x K
        :return: value (B, M, (T-1), 1), gradient_x (B, M, (T-1), D)
        """
        t, x, dw, u = data
        N = self.eqn_config.time_steps
        y = self.g_tf(x, u) # (B, M, 1)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            f = self.net_forward((t, x, u))
            grad = tape.gradient(f, x) # grad wrt whole Markovian variable (X_t, M_t) 
                                            # First self.dim entry is the grad w.r.t X_t
        z = self.z_tf(t, x, grad, dw, u) # (B, M, N, 1)
        y_values = tf.TensorArray(tf.float32, size=self.config.time_steps)
        y_values = y_values.write(N-1, y)
        for n in reversed(range(N-1)):
            y = y - self.h_tf(t[:,:,n+1,:], x[:,:,n+1,:], y, u[:,:,n+1,:]) - z[:,:,n,:]
            if n in self.exer_index:
                y = tf.maximum(y, self.option.early_payoff(x[:,:,n,:], u[:,:,n,:]))
            y_values = y_values.write(n, y)
        y_values = y_values.stack()
        y_values = tf.transpose(y_values, perm=[1, 2, 0, 3])
        loss_var = tf.reduce_mean(tf.math.reduce_variance(y, axis=1))
        loss_interior = tf.reduce_sum((f - y_values)**2, axis=2)
        loss = self.alpha * loss_var + loss_interior
        return loss, loss_interior, loss_var

    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor]) -> dict:
        with tf.GradientTape() as tape:
            with tf.name_scope('calling_model'):
                loss, loss_interior, loss_var = self(inputs[0])
            loss, loss_interior, loss_var = tf.reduce_mean(loss), tf.reduce_mean(loss_int), tf.reduce_mean(loss_tml)
            grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {"loss": loss,
                "loss interior": loss_interior,
                "loss var": loss_var}
    
    def h_tf(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:  # get h function
        """
        the driver term is r*y for MTG property
        t: [B, M, 1]
        x: [B, M, d]
        u_hat: [B, M, k]
        """
        if not isinstance(self.sde, TimeDependentGBM):
            r = tf.expand_dims(u_hat[:, :, 0], -1) # [B, M, 1]
        else:
            t = tf.expand_dims(t, axis=2)
            x = tf.expand_dims(x, axis=2)
            u_hat = tf.expand_dims(u_hat, axis=2)
            r = self.sde.drift_onestep(t, x, u_hat) # [B, M, 1]
            r = tf.squeeze(r, axis=2)
        return r * y

    def z_tf(self, t: tf.Tensor, x: tf.Tensor, grad: tf.Tensor, dw: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        grad: (B, M, N-1, d)
        dw: (B, M, N-1, d)
        give: \sigma(t, x) * grad
        for a batch of (t, x, par)
        """
        # v_tx = self.sde.diffusion_onestep(t, x, u_hat)
        if not isinstance(self.sde, HestonModel):
            x = x[...,:self.dim]
            grad = grad[...,:self.dim]
            v_tx = self.sde.diffusion_onestep(t, x[...,:self.dim], u_hat)
        else:
            v_tx = self.sde.diffusion_onestep(t, x, u_hat)
        z = tf.reduce_sum(v_tx * grad * dw, axis=-1, keepdims=True)
        return z

    def g_tf(self, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        payoff = self.option.payoff(x, u_hat)
        return payoff


class BSDEModel:
    def __init__(self, solver: BaseBSDESolver, sde: ItoProcessDriver, option: BaseOption, config):
        self.solver = solver
        self.sde = sde
        self.config = config # whole config
        self.option = option

    def pre_setting(self):
        learning_rate = self.config.net_config.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=200,
            decay_rate=0.9
        )
        Optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
        self.model = self.solver(self.sde, self.option, self.config)
        self.data_generator = DiffusionModelGenerator(self.sde, self.config, self.option, 100)
        # self.val_generator = DiffusionModelGenerator(self.sde, self.config, self.option, 20)
        self.model.compile(optimizer=Optimizer)

    def train(self, nr_epochs: int, checkpoint_path: str):
        self.pre_setting()
        # Create a callback that saves the model's batch loss
        class LossHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []
            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
        history = LossHistory()
        self.model.fit(x=self.data_generator, epochs=nr_epochs, callbacks=[history])
        self.model.no_net.save_weights(checkpoint_path)
        # return history
        return history
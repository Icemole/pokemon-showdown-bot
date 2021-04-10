from tensorflow.keras.models import clone_model
from rl.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np

# Class to pass as a callback to the training
# if you don't want the model to be updated
# at every step (i.e. passed by reference)
class ModelClonerCallback(Callback):
    num_eps_without_change = 0
    num_eps_before_change = 0
    player = None
    model = None

    def __init__(self, player, model, num_eps_before_change = 64):
        self.player = player
        self.model = model
        self.num_eps_before_change = num_eps_before_change

    def on_train_begin(self, logs = {}):
        self.player.model = clone_model(self.model.model)

    def on_episode_end(self, episode, logs = {}):
        self.num_eps_without_change += 1
        # self.model.summary() # does not work
        # self.model.model.summary() # works
        if self.num_eps_without_change == self.num_eps_before_change:
            self.num_eps_without_change = 0
            self.player.model = clone_model(self.model.model)


class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_cycle,
                 lr_decay=1,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.step_since_restart = 0
        self.steps_per_cycle = steps_per_cycle

        self.cycle_length = 1
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.step_since_restart / (self.steps_per_cycle * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        print('#'*100)
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_episode_end(self, episode, logs={}):
        self.step_since_restart += 1
        K.set_value(self.model.model.optimizer.lr, self.clr())

        if episode % self.steps_per_cycle == 0:
            self.step_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.model.set_weights(self.best_weights)

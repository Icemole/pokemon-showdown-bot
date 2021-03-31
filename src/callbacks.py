from tensorflow.keras.models import clone_model
from rl.callbacks import Callback

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

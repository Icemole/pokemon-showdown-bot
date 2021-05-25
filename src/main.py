## Tensorflow/Keras imports
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler as LRS
## Keras-RL2 imports
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
## Poke-env imports
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
## Extra imports
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from players import SimpleRLPlayer, CompleteInformationRLPlayer, SelfPlayRLPlayer, MaxDamagePlayer, MaxDamageTypedPlayer
from callbacks import ModelClonerCallback, SGDRScheduler
import argparse
import pickle


## Global variables
battle_format = "gen1randombattle"
env_player = SimpleRLPlayer(battle_format = battle_format,
        player_configuration=PlayerConfiguration(username="thisbotrules", password=None))
num_actions = len(env_player.action_space)

## Agent parameters
num_episodes = 100000


## Defines the agent's decision making model
def define_model():
    # input_layer = Input(shape=(1, env_player.num_features))
    # flatten_layer = Flatten()(input_layer)
    # moves_layer = Lambda(lambda x: x[:, :8])(flatten_layer)
    # moves_layer = Dense(16, activation="relu")(moves_layer)
    # moves_layer = Dense(16, activation="relu")(moves_layer)

    # remaining_team_layer = Lambda(lambda x: x[:, 8:])(flatten_layer)
    # remaining_team_layer = Dense(4, activation="relu")(remaining_team_layer)
    # remaining_team_layer = Dense(4, activation="relu")(remaining_team_layer)

    # multisource_model = Concatenate()([moves_layer, remaining_team_layer])
    # multisource_model = Dense(96, activation="relu")(multisource_model)
    # multisource_model = Dense(64, activation="relu")(multisource_model)
    # output_layer = Dense(num_actions, activation="linear")(multisource_model)

    # model = Model(input_layer, output_layer)

    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(1, env_player.num_features,)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_actions, activation="linear"))

    return model


## Defines the agent that will play the matches
## Returns a compiled agent and a self player
def define_agent(model, model_path):
    if model_path:
        model = keras.models.load_model(model_path)
    elif not model:
        model = define_model()
    # model.summary()
    memory = SequentialMemory(limit = 10000, window_length = 1)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000
    )
    dqn = DQNAgent(
        model=model,
        nb_actions=num_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=500,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(lr = 0.00025), metrics = ["mse"])

    return dqn


## Implements LR scheduling
def lr_scheduling(episode):
    if episode < num_episodes / 3:
        return 1e-3
    elif episode < 2 * num_episodes / 3:
        return 1e-4
    else:
        return 1e-5


## Trains the agent
def dqn_training(player, dqn, nb_steps, img_name, callbacks = []):
    training_history = dqn.fit(player, nb_steps = nb_steps, callbacks = callbacks)
    player.complete_current_battle()

    x = training_history.history["nb_steps"]
    y = training_history.history["episode_reward"]

    MAX_EPISODES_INCREMENTAL_MEAN = 1000
    avg_y = []
    avg_y_window = [0] * MAX_EPISODES_INCREMENTAL_MEAN
    avg_current = 0
    i = 0
    filled_list = False
    for val in y:
        # Incremental mean calculation
        if not filled_list:
            avg_current = (val + avg_current * i) / (i + 1)
        else:
            avg_current += (val - avg_y_window[i]) / MAX_EPISODES_INCREMENTAL_MEAN
        avg_y_window[i] = val
        i = (i + 1) % MAX_EPISODES_INCREMENTAL_MEAN
        if i == 0:
            filled_list = True
        avg_y.append(avg_current)
    # plt.plot(x, y, label="Recompensa")
    plt.plot(x, avg_y, label="Recompensa media (1000 combates)")
    plt.legend()
    plt.savefig(img_name)
    # with open("base_result.obj", "wb") as f:
    #     pickle.dump(training_history, f)


## Evaluates the agent
def dqn_evaluation(player, dqn, nb_episodes, callbacks = []):
    player.reset_battles()
    dqn.test(
        player,
        nb_episodes=nb_episodes,
        visualize=True,
        verbose=False
    )
    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_load", default=None,
            help="Path to a saved model.")
    parser.add_argument("--model_path_save", default="models/default_model_save",
            help="Path to save a model to.")
    args = parser.parse_args()

    ## Define the agent
    dqn = define_agent(env_player.default_model(multisource=False), args.model_path_load)

    ## Define the opponent
    # self_player = SelfPlayRLPlayer(dqn.model)
    max_damage_opponent = MaxDamagePlayer(battle_format = battle_format,
            player_configuration=PlayerConfiguration(username="opponentbot", password=None))
    #random_opponent = RandomPlayer(battle_format = battle_format)

    # Pre-training against max-damage opponent
    dqn.model.optimizer.lr = 1e-3
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=max_damage_opponent,
        env_algorithm_kwargs={
            "dqn": dqn,
            "nb_steps": num_episodes,
            "img_name": args.model_path_save + ".png"
        }
    )

    # SGDR Scheduler for self-play
    sgdr_lr = SGDRScheduler(min_lr=1e-4,
                             max_lr=1e-2,
                             steps_per_cycle=num_episodes/4,
                             lr_decay=0.95,
                             mult_factor=1.0)
    """
    # Self-play, num_eps_before_change should match steps_per_cycle or be a multiple of it to sync weight update with lr reset
    env_player.play_against(
        env_algorithm = dqn_training,
        opponent = self_player,
        env_algorithm_kwargs = {
            "dqn": dqn,
            "nb_steps": num_episodes,
            "callbacks": [ModelClonerCallback(self_player, dqn.model, num_eps_before_change = num_episodes/2), sgdr_lr]
        }
    )
    """

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=max_damage_opponent,
        env_algorithm_kwargs={
            "dqn": dqn,
            "nb_episodes": 1000,
            "callbacks": []
        }
    )

    # dqn.save_weights("../models/basic_selfplay_100k.h5f", overwrite = True)
    dqn.model.save(args.model_path_save)


if __name__ == "__main__":
    main()

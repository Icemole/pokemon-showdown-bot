## Tensorflow/Keras imports
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
## Keras-RL2 imports
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
## Poke-env imports
from poke_env.player.random_player import RandomPlayer
## Own imports
from players import SimpleRLPlayer, MaxDamagePlayer


## Global variables
battle_format = "gen1randombattle"
env_player = SimpleRLPlayer(battle_format = battle_format)
env_player2 = SimpleRLPlayer(battle_format = battle_format)
num_actions = len(env_player.action_space)


## Defines the agent's decision making model
def define_model():
    model = Sequential()
    model.add(Dense(128, activation = "relu", input_shape = (1, env_player.num_features,)))
    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(Dense(num_actions, activation = "linear", use_bias = False))

    return model


## Defines the agent that will play the matches
## Returns a compiled agent
def define_agent():
    model = define_model()
    # model.summary()
    memory = SequentialMemory(limit = 10000, window_length = 1)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr = "eps",
        value_max = 1.0,
        value_min = 0.05,
        value_test = 0,
        nb_steps = 10000
    )
    dqn = DQNAgent(
        model = model,
        nb_actions = num_actions,
        policy = policy,
        memory = memory,
        nb_steps_warmup = 1000,
        gamma = 0.5,
        target_model_update = 1,
        delta_clip = 0.01,
        enable_double_dqn = True,
    )
    dqn.compile(Adam(lr = 0.001), metrics = ["mse"])

    return dqn


## Trains the agent
def dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps = nb_steps)
    player.complete_current_battle()


## Evaluates the agent
def dqn_evaluation(player, dqn, nb_episodes):
    player.reset_battles()
    dqn.test(
        player,
        nb_episodes = nb_episodes,
        visualize = True,
        verbose = False
    )
    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


def main():
    ## Define the agent
    dqn = define_agent()

    ## Define the opponent
    opponent = RandomPlayer(battle_format = battle_format)
    better_opponent = MaxDamagePlayer(battle_format = battle_format)

    env_player.play_against(
        env_algorithm = dqn_training,
        opponent = opponent,
        env_algorithm_kwargs = {
            "dqn": dqn,
            "nb_steps": 50000,
        }
    )
    env_player.play_against(
        env_algorithm = dqqn_training,
        opponent = better_opponent,
        env_algorithm_kwargs = {
            "dqn": dqn,
            "nb_steps": 100000,
        }
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm = dqn_evaluation,
        opponent = better_opponent,
        env_algorithm_kwargs = {
            "dqn": dqn,
            "nb_episodes": 100,
        }
    )

    dqn.save_weights("../models/basic.h5f", overwrite = True)


if __name__ == "__main__":
    main()

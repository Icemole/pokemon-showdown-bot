# -*- coding: utf-8 -*-
from poke_env.player.env_player import Gen4EnvSinglePlayer
from poke_env.player.player import Player
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, Concatenate
from tensorflow.keras.models import Sequential, Model
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data import POKEDEX
import numpy as np


class Gen1EnvSinglePlayer(Gen4EnvSinglePlayer):
    # Follow the nomenclature of the source code
    _DEFAULT_BATTLE_FORMAT = "gen1randombattle"


class SimpleRLPlayer(Gen1EnvSinglePlayer):
    num_features = 10
    # Rewards
    fainted_reward =6.25
    victory_reward = 50
    # [BRN, FNT, FRZ, PAR, PSN, SLP, TOX]
    status_rewards = [0, 0, 0, 0, 0, 0, 0]

    def embed_battle(self, battle):
        """
        Calculates embed vector for the current moment in the battle
        Embed_battle vector:
        [active pokemon base power for all 4 moves] + [active pokemon damage multiplier for all 4 moves]
        + number of own remaining pokemon + number of enemy remaining pokemon
        :param battle:
        :return:
        """
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        remaining_mon_opponent = (
                len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [moves_base_power, moves_dmg_multiplier, [remaining_mon_team, remaining_mon_opponent]]
        )

    def compute_reward(self, battle) -> float:
        base_reward = self.reward_computing_helper(
            battle,
            fainted_value=self.fainted_reward,
            hp_value=1,
            victory_value=self.victory_reward,
        )
        # Does not take into account own status changes
        status_reward = 0
        for enemy in battle.opponent_team.values():
            if enemy.status is not None:
                status_reward += self.status_rewards[enemy.status.value - 1]
        # for ally in battle.team.values():
        #     if ally.status is not None:
        #         status_reward -= self.status_rewards[ally.status.value - 1]

        return base_reward + status_reward

    def default_model(self, multisource=False):
        if multisource:
            input_layer = Input(shape=(1, self.num_features))
            flatten_layer = Flatten()(input_layer)
            moves_layer = Lambda(lambda x: x[:, :8])(flatten_layer)
            moves_layer = Dense(16, activation="relu")(moves_layer)
            moves_layer = Dense(16, activation="relu")(moves_layer)

            remaining_team_layer = Lambda(lambda x: x[:, 8:])(flatten_layer)
            remaining_team_layer = Dense(4, activation="relu")(remaining_team_layer)
            remaining_team_layer = Dense(4, activation="relu")(remaining_team_layer)

            multisource_model = Concatenate()([moves_layer, remaining_team_layer])
            multisource_model = Dense(96, activation="relu")(multisource_model)
            multisource_model = Dense(64, activation="relu")(multisource_model)
            output_layer = Dense(len(self.action_space), activation="linear")(multisource_model)

            model = Model(input_layer, output_layer)

        else:
            model = Sequential()
            model.add(Dense(128, activation="relu", input_shape=(1, self.num_features,)))
            model.add(Flatten())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(len(self.action_space), activation="linear"))

        return model


class IdRLPlayer(Gen1EnvSinglePlayer):
    num_features = 20
    # Rewards
    fainted_reward = 6.25
    victory_reward = 50
    # [BRN, FNT, FRZ, PAR, PSN, SLP, TOX]
    status_rewards = [0,0,0,0,0,0,0]

    def embed_battle(self, battle):
        """
        Calculates embed vector for the current moment in the battle
        Embed_battle vector:
        [active pokemon base power for all 4 moves] + [active pokemon damage multiplier for all 4 moves]
        + id of active pokemon + id of enemy active pokemon
        + [ids of own remaining pokemons] + [id of enemy remaining pokemon]
        :param battle:
        :return:
        """
        # -1 indicates that the move does not have a base power or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100 # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
                # take same type attack bonus into consideration
                if moves_dmg_multiplier[i] != 0:
                    if move.type == battle.active_pokemon.type_1 or move.type == battle.active_pokemon.type_2:
                        moves_dmg_multiplier[i] += 0.5

        # Dex number for active own and rival Pokemon
        active_mon = POKEDEX[battle.active_pokemon.species].get("num", -1)
        active_mon_opponent = POKEDEX[battle.opponent_active_pokemon.species].get("num", -1)

        # Get dex numbers of remaining own Pokemon available to switch, or -1 if the Pokemon has fainted
        remaining_mon_team = [POKEDEX[mon.species].get("num") if not mon.fainted else -1
                              for identifier, mon in battle.team.items()
                              if POKEDEX[mon.species].get("num") != active_mon]
        # Get dex numbers of Rival Pokemon available to switch, -1 if the Pokemon has fainted or -2 if it is unknown (has not been seen yet)
        remaining_mon_opponent = [POKEDEX[mon.species].get("num") if not mon.fainted else -1
                              for identifier, mon in battle.opponent_team.items()
                              if POKEDEX[mon.species].get("num") != active_mon_opponent]
        remaining_mon_opponent += [-2] * (5 - len(remaining_mon_opponent))

        # Final vector with 20 components
        return np.concatenate(
            [moves_base_power, moves_dmg_multiplier, [active_mon, active_mon_opponent], remaining_mon_team,
             remaining_mon_opponent]
        )

    def default_model(self, multisource=False):
        if multisource:
            input_layer = Input(shape=(1, self.num_features))
            flatten_layer = Flatten()(input_layer)
            moves_layer = Lambda(lambda x: x[:, :8])(flatten_layer)
            moves_layer = Dense(16, activation="relu")(moves_layer)
            moves_layer = Dense(16, activation="relu")(moves_layer)

            remaining_team_layer = Lambda(lambda x: x[:, 8:])(flatten_layer)
            remaining_team_layer = Dense(64, activation="relu")(remaining_team_layer)
            remaining_team_layer = Dense(32, activation="relu")(remaining_team_layer)
            remaining_team_layer = Dense(16, activation="relu")(remaining_team_layer)
            remaining_team_layer = Dense(16, activation="relu")(remaining_team_layer)

            multisource_model = Concatenate()([moves_layer, remaining_team_layer])
            multisource_model = Dense(96, activation="relu")(multisource_model)
            multisource_model = Dense(64, activation="relu")(multisource_model)
            output_layer = Dense(len(self.action_space), activation="linear")(multisource_model)

            model = Model(input_layer, output_layer)

        else:
            model = Sequential()
            model.add(Dense(128, activation="relu", input_shape=(1, self.num_features,)))
            model.add(Flatten())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(len(self.action_space), activation="linear"))

        return model

class CompleteInformationRLPlayer(Gen1EnvSinglePlayer):
    num_features = 34
    # Rewards
    fainted_reward = 6.25
    victory_reward = 50
    # [BRN, FNT, FRZ, PAR, PSN, SLP, TOX]
    status_rewards = [0, 0, 0, 0, 0, 0, 0]

    def embed_battle(self, battle):
        """
        Calculates embed vector for the current moment in the battle
        Embed_battle vector:
        [active pokemon base power for all 4 moves] + [active pokemon damage multiplier for all 4 moves]
        + [base stats of active pokemon] + [base stats of enemy active pokemon]
        + id of active pokemon + id of enemy active pokemon
        + [ids of own remaining pokemons] + [id of enemy remaining pokemon]
        :param battle:
        :return:
        """
        # -1 indicates that the move does not have a base power or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
                # take same type attack bonus into consideration
                if moves_dmg_multiplier[i] != 0:
                    if move.type == battle.active_pokemon.type_1 or move.type == battle.active_pokemon.type_2:
                        moves_dmg_multiplier[i] += 0.5

        # Dex number for active own and rival Pokemon
        active_mon = POKEDEX[battle.active_pokemon.species].get("num", -1)
        active_mon_opponent = POKEDEX[battle.opponent_active_pokemon.species].get("num", -1)

        # Get dex numbers of remaining own Pokemon available to switch, or -1 if the Pokemon has fainted
        remaining_mon_team = [POKEDEX[mon.species].get("num") if not mon.fainted else -1
                              for identifier, mon in battle.team.items()
                              if POKEDEX[mon.species].get("num") != active_mon]
        # Get dex numbers of Rival Pokemon available to switch, -1 if the Pokemon has fainted or -2 if it is unknown (has not been seen yet)
        remaining_mon_opponent = [POKEDEX[mon.species].get("num") if not mon.fainted else -1
                                  for identifier, mon in battle.opponent_team.items()
                                  if POKEDEX[mon.species].get("num") != active_mon_opponent]
        remaining_mon_opponent += [-2] * (5 - len(remaining_mon_opponent))

        # Final vector with 20 components
        f = [moves_base_power, moves_dmg_multiplier,
             [battle.active_pokemon.current_hp], battle.active_pokemon.base_stats.values(),
             [battle.opponent_active_pokemon.current_hp], battle.opponent_active_pokemon.base_stats.values(),
             [active_mon, active_mon_opponent],
             remaining_mon_team, remaining_mon_opponent]
        return np.concatenate(
            [moves_base_power, moves_dmg_multiplier,
             [battle.active_pokemon.current_hp], list(battle.active_pokemon.base_stats.values()),
             [battle.opponent_active_pokemon.current_hp], list(battle.opponent_active_pokemon.base_stats.values()),
             [active_mon, active_mon_opponent],
             remaining_mon_team, remaining_mon_opponent]
        )


    def compute_reward(self, battle) -> float:
        base_reward = self.reward_computing_helper(
            battle,
            fainted_value = self.fainted_reward,
            hp_value = 1,
            victory_value = self.victory_reward,
        )
        # Does not take into account own status changes
        status_reward = 0
        for enemy in battle.opponent_team.values():
            if enemy.status is not None:
                status_reward += self.status_rewards[enemy.status.value - 1]
        # for ally in battle.team.values():
        #     if ally.status is not None:
        #         status_reward -= self.status_rewards[ally.status.value - 1]
        
        return base_reward + status_reward

    def default_model(self, multisource=False):
        if multisource:
            input_layer = Input(shape=(1, self.num_features))
            flatten_layer = Flatten()(input_layer)
            moves_layer = Lambda(lambda x: x[:, :8])(flatten_layer)
            moves_layer = Dense(16, activation="relu")(moves_layer)
            moves_layer = Dense(16, activation="relu")(moves_layer)

            stats_layer = Lambda(lambda x: x[:, 8:24])(flatten_layer)
            stats_layer = Dense(32, activation="relu")(stats_layer)
            stats_layer = Dense(16, activation="relu")(stats_layer)

            remaining_team_layer = Lambda(lambda x: x[:, :24])(flatten_layer)
            remaining_team_layer = Dense(64, activation="relu")(remaining_team_layer)
            remaining_team_layer = Dense(32, activation="relu")(remaining_team_layer)
            remaining_team_layer = Dense(16, activation="relu")(remaining_team_layer)
            remaining_team_layer = Dense(16, activation="relu")(remaining_team_layer)

            multisource_model = Concatenate()([moves_layer, stats_layer, remaining_team_layer])
            multisource_model = Dense(96, activation="relu")(multisource_model)
            multisource_model = Dense(64, activation="relu")(multisource_model)
            output_layer = Dense(len(self.action_space), activation="linear")(multisource_model)

            model = Model(input_layer, output_layer)

        else:
            model = Sequential()
            model.add(Dense(128, activation="relu", input_shape=(1, self.num_features,)))
            model.add(Flatten())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(len(self.action_space), activation="linear"))

        return model


class SelfPlayRLPlayer(SimpleRLPlayer):
    model = None

    # clone_model: 
    def __init__(self, model):
        self.model = model
        super().__init__(battle_format = "gen1randombattle")

    def choose_move(self, battle):
        state = super().embed_battle(battle)
        # I don't like this
        # Need to expand the vector twice to reach dimension 3
        state = np.expand_dims(state, axis = 0)
        state = np.expand_dims(state, axis = 0)
        predictions = self.model.predict(state)
        action = np.argmax(predictions)

        return super()._action_to_move(action, battle)
    
    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        # self._observations[battle].put(self.embed_battle(battle))
        pass


class OnlineRLPlayer(SimpleRLPlayer):

    def set_model(self, model):
        self.model = model

    def choose_move(self, battle):
        if hasattr(self, 'model'):
            state = super().embed_battle(battle)
            # I don't like this
            # Need to expand the vector twice to reach dimension 3
            state = np.expand_dims(state, axis = 0)
            state = np.expand_dims(state, axis = 0)
            predictions = self.model.predict(state)
            action = np.argmax(predictions)
            return super()._action_to_move(action, battle)
        else:
            # if no model is set, fall back on using a random move
            return self.choose_random_move(battle)


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


class MaxDamageTypedPlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            moves_dmg = []
            for i, move in enumerate(battle.available_moves):
                if move.type:
                    dmg_multiplier = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                    )
                    # take same type attack bonus into consideration
                    if dmg_multiplier != 0:
                        if move.type == battle.active_pokemon.type_1 or move.type == battle.active_pokemon.type_2:
                            dmg_multiplier += 0.5
                    moves_dmg.append(dmg_multiplier * move.base_power)
                else:
                    moves_dmg.append(move.base_power)
            posmov = moves_dmg.index(max(moves_dmg))
            best_move = battle.available_moves[posmov]

            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

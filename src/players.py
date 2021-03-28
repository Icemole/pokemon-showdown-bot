# -*- coding: utf-8 -*-
from poke_env.player.env_player import Gen4EnvSinglePlayer
from poke_env.player.player import Player
import numpy as np


class Gen1EnvSinglePlayer(Gen4EnvSinglePlayer):
    # Follow the nomenclature of the source code
    _DEFAULT_BATTLE_FORMAT = "gen1randombattle"


class SimpleRLPlayer(Gen1EnvSinglePlayer):
    num_features = 10
    # Rewards
    fainted_reward = 2
    victory_reward = 50
    # [BRN, FNT, FRZ, PAR, PSN, SLP, TOX]
    status_rewards = [0.1, 0, 0.2, 0.2, 0.1, 0.2, 0]

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100 # Simple rescaling to facilitate learning
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

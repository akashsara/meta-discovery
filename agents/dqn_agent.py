# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
import numpy as np
import torch

from poke_env.player.env_player import Gen8EnvSinglePlayer


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, model=None, *args, **kwargs):
        Gen8EnvSinglePlayer.__init__(self, *args, **kwargs)
        self.model = model

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        self.mask = self.make_mask(battle)

        # Final vector with 10 components
        return torch.cat(
            [
                torch.tensor(moves_base_power),
                torch.tensor(moves_dmg_multiplier),
                torch.tensor([remaining_mon_team, remaining_mon_opponent]),
            ], dim=-1
        ).float()

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )

    def get_action_mask(self):
        return self.mask

    def make_mask(self, battle):
        mask = []
        for action in range(len(self.action_space)):
            if (
                action < 4
                and action < len(battle.available_moves)
                and not battle.force_switch
            ):
                mask.append(0)
            elif (
                not battle.force_switch
                and battle.can_z_move
                and battle.active_pokemon
                and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
            ):
                mask.append(0)
            elif (
                battle.can_mega_evolve
                and 0 <= action - 8 < len(battle.available_moves)
                and not battle.force_switch
            ):
                mask.append(0)
            elif (
                battle.can_dynamax
                and 0 <= action - 12 < len(battle.available_moves)
                and not battle.force_switch
            ):
                mask.append(0)
            elif 0 <= action - 16 < len(battle.available_switches):
                mask.append(0)
            else:
                mask.append(-1e9)
        return torch.tensor(mask).float()


class SimpleRLPlayerTesting(SimpleRLPlayer):
    def __init__(self, model, *args, **kwargs):
        SimpleRLPlayer.__init__(self, *args, **kwargs)
        self.model = model

    def choose_move(self, battle):
        state = self.embed_battle(battle).reshape(1, 1, -1)
        predictions = self.model.predict(state)[0]
        action = np.argmax(predictions)
        return self._action_to_move(action, battle)

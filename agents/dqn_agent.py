# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
import numpy as np
import torch

import sys
sys.path.append("./")
from agents.env_player import Gen8EnvSinglePlayerFixed
from poke_env.player.env_player import Gen8EnvSinglePlayer


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayerFixed):
    def __init__(self, model=None, *args, **kwargs):
        super(Gen8EnvSinglePlayerFixed, self).__init__(*args, **kwargs)
        self.model = model

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(list(battle.active_pokemon.moves.values())):
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

    def skip_current_step(self):
        return self.skip_step

    def make_mask(self, battle):
        self.skip_step = False
        mask = []
        moves = list(battle.active_pokemon.moves.values())
        team = list(battle.team.values())
        force_switch = (len(battle.available_switches) > 0) and battle.force_switch
        available_move_ids = [x.id for x in battle.available_moves]
        available_z_moves = [x.id for x in battle.active_pokemon.available_z_moves]

        for action in range(len(self.action_space)):
            if (
                action < 4
                and action < len(moves)
                and not force_switch
                and moves[action].id in available_move_ids
            ):
                mask.append(0)
            elif (
                battle.can_z_move
                and battle.active_pokemon
                and 0 <= action - 4 < len(moves)
                and not force_switch
                and moves[action - 4].id in available_z_moves
            ):
                mask.append(0)
            elif (
                battle.can_mega_evolve
                and 0 <= action - 8 < len(moves)
                and not force_switch
                and moves[action - 8].id in available_move_ids
            ):
                mask.append(0)
            elif (
                battle.can_dynamax
                and 0 <= action - 12 < len(moves)
                and not force_switch
                and moves[action - 12].id in available_move_ids
            ):
                mask.append(0)
            elif (
                not battle.trapped
                and 0 <= action - 16 < len(team)
                and team[action - 16] in battle.available_switches
            ):
                mask.append(0)
            else:
                mask.append(-1e9)
        # Special case for Struggle since it doesn't show up in battle.moves
        if (
            len(battle.available_moves) == 1
            and battle.available_moves[0].id == "struggle"
            and not force_switch
        ):
            mask[0] = 0

        # Special case for the buggy scenario where there are 
        # no available moves nor switches
        # Ref: https://github.com/hsahovic/poke-env/issues/295
        # But also a generic catch-all for scenarios with no moves
        if all([x == -1e9 for x in mask]):
            self.skip_step = True
            print("ALL MASKED")
            print(mask)
            print(battle.won, battle.lost, battle.finished, battle.turn)
            print(battle.force_switch, force_switch, battle.trapped)
            print(battle.active_pokemon)
            print(battle.team)
            print(battle.available_switches)
            print(battle.available_moves)
            print(battle.active_pokemon.moves)
        
        if len(battle.available_moves) == 0 and len(battle.available_switches) == 0:
            print("NO ACTIONS AVAILABLE")
            print(mask)
            print(battle.won, battle.lost, battle.finished, battle.turn)
            print(battle.force_switch, force_switch, battle.trapped)
            print(battle.active_pokemon)
            print(battle.team)
            print(battle.available_switches)
            print(battle.available_moves)
            print(battle.active_pokemon.moves)

        if not force_switch and battle.force_switch:
            print("FORCE SWITCH WITH NO SWITCHES")
            print(mask)
            print(battle.won, battle.lost, battle.finished, battle.turn)
            print(battle.force_switch, force_switch, battle.trapped)
            print(battle.active_pokemon)
            print(battle.team)
            print(battle.available_switches)
            print(battle.available_moves)
            print(battle.active_pokemon.moves)

        if (len(battle.available_moves) == 0 and len(battle.available_switches) == 0) or all([x == -1e9 for x in mask]) or (not force_switch and battle.force_switch):
            print("###" * 30)
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

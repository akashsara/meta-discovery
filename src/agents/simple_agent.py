# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
import numpy as np
import torch

import sys

sys.path.append("./")
from agents.env_player import Gen8EnvSinglePlayerFixed
from gym.spaces import Space, Box
from poke_env.player.player import Player
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

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
            len([mon for mon in battle.team.values() if not mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if not mon.fainted]) / 6
        )

        self.mask = self.make_mask(battle)

        # Final vector with 10 components
        return torch.cat(
            [
                torch.tensor(moves_base_power),
                torch.tensor(moves_dmg_multiplier),
                torch.tensor([remaining_mon_team, remaining_mon_opponent]),
            ],
            dim=-1,
        ).float()

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2, hp_value=1, victory_value=30
        )

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def action_masks(self):
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

        for action in range(self.action_space.n):
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
        return torch.tensor(mask).float()


class SimpleRLPlayerTesting(SimpleRLPlayer):
    def __init__(self, model, *args, **kwargs):
        SimpleRLPlayer.__init__(self, *args, **kwargs)
        self.model = model


class GeneralAPISimpleAgent(Player):
    def __init__(self, model, device, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        self.model = model
        self.device = device
        self.is_actor_critic = "ActorCritic" in str(model)
        # NOTE: This is subject to change based on the current generation.
        # Hard-coding it for now but this might cause issues later on.
        self.action_space = list(range(4 * 4 + 6))

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        if battle.active_pokemon:
            available_moves = list(battle.active_pokemon.moves.values())
        else:
            available_moves = []

        for i, move in enumerate(available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type and battle.opponent_active_pokemon is not None:
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
            ],
            dim=-1,
        ).float()

    def make_mask(self, battle):
        self.skip_step = False
        mask = []
        if battle.active_pokemon:
            moves = list(battle.active_pokemon.moves.values())
            available_z_moves = [x.id for x in battle.active_pokemon.available_z_moves]
        else:
            moves = []
            available_z_moves = []
        team = list(battle.team.values())
        force_switch = (len(battle.available_switches) > 0) and battle.force_switch
        available_move_ids = [x.id for x in battle.available_moves]

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
        return torch.tensor(mask).float()

    def action_masks(self):
        return self.mask

    def skip_current_step(self):
        return self.skip_step

    def action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        """Converts actions to move orders.

        The conversion is done as follows:

        action = -1:
            The battle will be forfeited.
        0 <= action < 4:
            The actionth available move in battle.active_pokemon.moves is
            executed.
        4 <= action < 8:
            The action - 4th available move in battle.active_pokemon.moves is
            executed, with z-move.
        8 <= action < 12:
            The action - 8th available move in battle.active_pokemon.moves is
            executed, with mega-evolution.
        12 <= action < 16:
            The action - 12th available move in battle.active_pokemon.moves is
            executed, while dynamaxing.
        16 <= action < 22
            The action - 16th available switch in battle.team is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if battle.active_pokemon:
            moves = list(battle.active_pokemon.moves.values())
            available_z_moves = [
                move.id for move in battle.active_pokemon.available_z_moves
            ]
        else:
            moves = []
            available_z_moves = []
        team = list(battle.team.values())
        force_switch = (len(battle.available_switches) > 0) and battle.force_switch
        # We use the ID of each move here since in some cases
        # The same moves might be in both battle.available_moves
        # and battle.active_pokemon.moves but they may not be the same object
        available_move_ids = [move.id for move in battle.available_moves]

        if action == -1:
            return ForfeitBattleOrder()
        # Special case for moves that are never a part of pokemon.moves
        # Example: Struggle, Locked into Outrage via Copycat
        elif (
            action < 4
            and action < len(moves)
            and not force_switch
            and len(available_move_ids) == 1
        ):
            return self.create_order(battle.available_moves[0])
        elif (
            action < 4
            and action < len(moves)
            and not force_switch
            and moves[action].id in available_move_ids
        ):
            return self.create_order(moves[action])
        elif (
            battle.can_z_move
            and battle.active_pokemon
            and 0 <= action - 4 < len(moves)
            and not force_switch
            and moves[action - 4].id in available_z_moves
        ):
            return self.create_order(moves[action - 4], z_move=True)
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(moves)
            and not force_switch
            and moves[action - 8].id in available_move_ids
        ):
            return self.create_order(moves[action - 8], mega=True)
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(moves)
            and not force_switch
            and moves[action - 12].id in available_move_ids
        ):
            return self.create_order(moves[action - 12], dynamax=True)
        elif (
            not battle.trapped
            and 0 <= action - 16 < len(team)
            and team[action - 16] in battle.available_switches
        ):
            return self.create_order(team[action - 16])
        else:
            return self.choose_random_move(battle)

    def get_action(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            if self.is_actor_critic:
                predictions, _ = self.model(state)
            else:
                predictions = self.model(state)
        return predictions.cpu()

    def choose_move(self, battle):
        state = self.embed_battle(battle)
        action_mask = self.action_masks()
        skip_model = self.skip_current_step()

        # Special case for the buggy scenario where there are
        # no available moves nor switches
        # Ref: https://github.com/hsahovic/poke-env/issues/295
        if skip_model:
            return self.choose_random_move(battle)
        else:
            predictions = self.get_action(state)
            action = np.argmax(predictions + action_mask)
            return self.action_to_move(action, battle)

# -*- coding: utf-8 -*-
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from poke_env.player.env_player import Gen8EnvSinglePlayer

class Gen8EnvSinglePlayerFixed(Gen8EnvSinglePlayer):
    """
    Fixes an issue with inconsistencies in the order of items
    in battle.team vs battle.available_switches
    and battle.active_pokemon.moves vs battle.available_moves
    Ref: https://github.com/hsahovic/poke-env/issues/292
    """

    def __init__(self, *args, **kwargs):
        super(Gen8EnvSinglePlayerFixed, self).__init__(*args, **kwargs)
        self.done_training = False
        self.all_wins = []
        self.all_losses = []

    def reset_statistics(self):
        self.all_wins = []
        self.all_losses = []

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
        moves = list(battle.active_pokemon.moves.values())
        team = list(battle.team.values())
        force_switch = (len(battle.available_switches) > 0) and battle.force_switch
        # We use the ID of each move here since in some cases
        # The same moves might be in both battle.available_moves
        # and battle.active_pokemon.moves but they may not be the same object
        available_move_ids = [move.id for move in battle.available_moves]
        available_z_moves = [move.id for move in battle.active_pokemon.available_z_moves]

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
            return self.agent.create_order(battle.available_moves[0])
        elif (
            action < 4
            and action < len(moves)
            and not force_switch
            and moves[action].id in available_move_ids
        ):
            return self.agent.create_order(moves[action])
        elif (
            battle.can_z_move
            and battle.active_pokemon
            and 0 <= action - 4 < len(moves)
            and not force_switch
            and moves[action - 4].id in available_z_moves
        ):
            return self.agent.create_order(moves[action - 4], z_move=True)
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(moves)
            and not force_switch
            and moves[action - 8].id in available_move_ids
        ):
            return self.agent.create_order(moves[action - 8], mega=True)
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(moves)
            and not force_switch
            and moves[action - 12].id in available_move_ids
        ):
            return self.agent.create_order(moves[action - 12], dynamax=True)
        elif (
            not battle.trapped
            and 0 <= action - 16 < len(team)
            and team[action - 16] in battle.available_switches
        ):
            return self.agent.create_order(team[action - 16])
        else:
            return self.agent.choose_random_move(battle)

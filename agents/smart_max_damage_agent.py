# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/max_damage_player.py
from poke_env.player.player import Player

class SmartMaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            best_move = None
            best_power = -1
            player = battle.active_pokemon
            opponent = battle.opponent_active_pokemon
            # Finds the best move among available ones
            for move in battle.available_moves:
                move_type = move.type
                stab_bonus = 1
                if move_type in player.types:
                    stab_bonus = 1.5
                base_power = move.base_power
                accuracy = move.accuracy
                expected_hits = move.expected_hits
                expected_power = accuracy * (base_power * expected_hits) * stab_bonus
                # Calculate type effectiveness
                type_effectiveness = opponent.damage_multiplier(move)
                expected_power *= type_effectiveness
                if expected_power > best_power:
                    best_move = move
                    best_power = expected_power
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
import numpy as np
import pandas as pd
import sys
import joblib

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.environment.weather import Weather
from poke_env.environment.field import Field
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.effect import Effect
from poke_env.environment.status import Status
from poke_env.environment.pokemon_type import PokemonType
from poke_env.utils import to_id_str
import torch

# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class FullStatePlayer(Gen8EnvSinglePlayer):
    def __init__(self, config, *args, **kwargs):
        """
        Some Notes:
        We create a lookup table for Pokemon/Abilities/Moves/Items.
        This is becauuse we need to learn embeddings for these.
        And the API does not have an easy way of obtaining the maximum value.
        There are some other variables that require a lookup as well.
        The full list is given below.
        Everything else is either a single integer or boolean.
        Embeddings: Pokemon, Move, Ability, Item
        One Hot Encoding: Weather, Field, SideCondition, Type, Effect, Status
        Normalized Values: Stats, Boosts

        Multi-Value Possible: SideCondition, Field, Effect
        SideCondition has +3 since Spikes/Toxic Spikes are stackable conditions.
        However the API returns it as SPIKES: <COUNT>. We convert these counts
        to separate values.
        """
        Gen8EnvSinglePlayer.__init__(self, *args, **kwargs)
        if config["create"]:
            # Pokedex
            df = pd.read_json(config["pokemon_json"]).T
            df["num"] = df["num"].astype(int)
            df.drop(df[df["num"] <= 0].index, inplace=True)
            pokemon = df.index.tolist() + ["unknown_pokemon"]
            # Cosmetic Formes - Special case where certain Pokemon have
            # different cosmetic forms. They have no gameplay impact though.
            # So we consider them as their base form.
            cosmetic = {
                x: y
                for x, y in df["cosmeticFormes"].fillna(False).to_dict().items()
                if y
            }
            cosmetic_lookup = {to_id_str(z): x for x, y in cosmetic.items() for z in y}
            # Stats
            stats = list(df["baseStats"][0].keys())
            # Abilities
            abilities = set(
                [to_id_str(y) for x in df["abilities"].tolist() for y in x.values()]
            )
            abilities = list(abilities) + ["unknown_ability"]
            # Moves
            df = pd.read_json(config["moves_json"]).T
            moves = df.index.tolist() + ["unknown_move"]

            # Boosts
            boosts = df["boosts"][~df["boosts"].isnull()].tolist()
            boosts = list(set([key for item in boosts for key in item]))

            # Items
            df = pd.read_json(config["items_json"])
            items = df.index.tolist() + ["unknown_item"]

            max_values = {
                "species": len(pokemon),
                "moves": len(moves),
                "abilities": len(abilities),
                "items": len(items),
                "stats": len(stats),
                "boosts": len(boosts),
                "types": len(PokemonType),
                "weather": len(Weather),
                "fields": len(Field),
                "side_conditions": len(SideCondition) + 3,
                "effects": len(Effect),
                "status": len(Status),
            }
            self.lookup = {
                "pokemon": {x: i for i, x in enumerate(pokemon)},
                "abilities": {x: i for i, x in enumerate(abilities)},
                "moves": {x: i for i, x in enumerate(moves)},
                "stats": {x: i for i, x in enumerate(stats)},
                "boosts": {x: i for i, x in enumerate(boosts)},
                "items": {x: i for i, x in enumerate(items)},
                "max_values": max_values,
                "cosmetic": cosmetic_lookup,
            }
            joblib.dump(self.lookup, config["lookup_filename"])
        else:
            self.lookup = joblib.load(config["lookup_filename"])
        self.state_length_dict = {}

    def create_empty_state_vector(self):
        pokemon = {
            "species": self.lookup["pokemon"]["unknown_pokemon"],
            "item": self.lookup["items"]["unknown_item"],
            "ability": self.lookup["abilities"]["unknown_ability"],
            "possible_abilities": [self.lookup["abilities"]["unknown_ability"]] * 3,
            "moves": [self.lookup["moves"]["unknown_move"]] * 4,
            "type": torch.zeros((self.lookup["max_values"]["types"])),
            "effects": torch.zeros((self.lookup["max_values"]["effects"])),
            "status": torch.zeros((self.lookup["max_values"]["status"])),
            # TODO: Set this to the average stat value instead.
            "base_stats": [0] * self.lookup["max_values"]["stats"],
            "stat_boosts": [0] * self.lookup["max_values"]["boosts"],
            "moves_pp": [1] * 4,
            "level": 0,
            "health": 1,
            "protect_counter": 0,
            "status_counter": 0,
            "fainted": False,
            "active": False,
            "first_turn": False,
            "must_recharge": False,
            "preparing": False,
        }
        return {
            "battle": {
                "weather": torch.zeros((self.lookup["max_values"]["weather"])),
                "fields": torch.zeros((self.lookup["max_values"]["fields"])),
            },
            "player_team": {
                "side_conditions": torch.zeros(
                    (self.lookup["max_values"]["side_conditions"])
                ),
                "can_dynamax": False,
                "dynamax_turns_left": 0,
                "pokemon": [pokemon for i in range(6)],
            },
            "opponent_team": {
                "side_conditions": torch.zeros(
                    (self.lookup["max_values"]["side_conditions"])
                ),
                "can_dynamax": False,
                "dynamax_turns_left": 0,
                "pokemon": [pokemon for i in range(6)],
            },
        }

    def state_to_machine_readable_state(self, state):
        battle_state = torch.cat(
            [state["battle"]["weather"], state["battle"]["fields"]]
        ).float()
        player_state = torch.cat(
            [
                state["player_team"]["side_conditions"],
                torch.tensor([state["player_team"]["can_dynamax"]]),
                torch.tensor([state["player_team"]["dynamax_turns_left"]]),
            ]
        ).float()
        opponent_state = torch.cat(
            [
                state["opponent_team"]["side_conditions"],
                torch.tensor([state["opponent_team"]["can_dynamax"]]),
                torch.tensor([state["opponent_team"]["dynamax_turns_left"]]),
            ]
        ).float()
        teams = {
            "player_team": [],
            "opponent_team": [],
        }
        for key, value in teams.items():
            for pokemon in state[key]["pokemon"]:
                pokemon_others_state = torch.cat(
                    [
                        pokemon["type"],
                        pokemon["effects"],
                        pokemon["status"],
                        torch.tensor(pokemon["base_stats"]),
                        torch.tensor(pokemon["stat_boosts"]),
                        torch.tensor(pokemon["moves_pp"]),
                        torch.tensor([pokemon["level"]]),
                        torch.tensor([pokemon["health"]]),
                        torch.tensor([pokemon["protect_counter"]]),
                        torch.tensor([pokemon["status_counter"]]),
                        torch.tensor([pokemon["fainted"]]),
                        torch.tensor([pokemon["active"]]),
                        torch.tensor([pokemon["first_turn"]]),
                        torch.tensor([pokemon["must_recharge"]]),
                        torch.tensor([pokemon["preparing"]]),
                    ]
                ).float()
                pokemon_state = torch.cat(
                    [
                        torch.tensor([pokemon["species"]]),
                        torch.tensor([pokemon["item"]]),
                        torch.tensor([pokemon["ability"]]),
                        *[torch.tensor([x]) for x in pokemon["possible_abilities"]],
                        *[torch.tensor([x]) for x in pokemon["moves"]],
                        pokemon_others_state,
                    ]
                )
                # Ensures active Pokemon is always in the first slot of the team
                if pokemon["active"]:
                    teams[key].insert(0, pokemon_state)
                else:
                    teams[key].append(pokemon_state)
            teams[key] = torch.cat(teams[key])
        return torch.cat(
            [
                teams["player_team"],
                player_state,
                teams["opponent_team"],
                opponent_state,
                battle_state,
            ]
        )

    def get_state_lengths(self):
        state = self.create_empty_state_vector()
        self.state_length_dict["battle_state"] = torch.cat(
            [state["battle"]["weather"], state["battle"]["fields"]]
        ).shape[0]
        self.state_length_dict["team_state"] = torch.cat(
            [
                state["player_team"]["side_conditions"],
                torch.tensor([state["player_team"]["can_dynamax"]]),
                torch.tensor([state["player_team"]["dynamax_turns_left"]]),
            ]
        ).shape[0]
        pokemon = state["player_team"]["pokemon"][0]
        self.state_length_dict["pokemon_state"] = torch.cat(
            [
                torch.tensor([pokemon["species"]]),
                torch.tensor([pokemon["item"]]),
                torch.tensor([pokemon["ability"]]),
                *[torch.tensor([x]) for x in pokemon["possible_abilities"]],
                *[torch.tensor([x]) for x in pokemon["moves"]],
            ]
        ).shape[0]
        self.state_length_dict["pokemon_others_state"] = torch.cat(
            [
                pokemon["type"],
                pokemon["effects"],
                pokemon["status"],
                torch.tensor(pokemon["base_stats"]),
                torch.tensor(pokemon["stat_boosts"]),
                torch.tensor(pokemon["moves_pp"]),
                torch.tensor([pokemon["level"]]),
                torch.tensor([pokemon["health"]]),
                torch.tensor([pokemon["protect_counter"]]),
                torch.tensor([pokemon["status_counter"]]),
                torch.tensor([pokemon["fainted"]]),
                torch.tensor([pokemon["active"]]),
                torch.tensor([pokemon["first_turn"]]),
                torch.tensor([pokemon["must_recharge"]]),
                torch.tensor([pokemon["preparing"]]),
            ]
        ).shape[0]
        return self.state_length_dict

    def embed_battle(self, battle):
        """
        Notes:
        1. We would ideally like to have a min turns remaining and a max turns
           remaining for things like fields, weather and side conditions. But
           there's no easy way of tracking that in the current iteration of
           poke_env.
        2. We only use base stats for both sides. While this does ignore things
           like EVs and IVs, it's the fairest option. While we can use stats
           for the player side of things, the final stats aren't available for
           the opponent side. metagrok does an estimation of stats by assuming
           certain values of EVs and IVs. Specifically, each Pokemon has a
           neutral nature and all EVs are split equally. However this
           assumption holds only for random battles.
        3. For Weather, Effect, Field, SideCondition, Status and Type
           We subtract 1 from the value returned by poke_env.
           This is because poke_env does not use zero indexing.
        """
        state = self.create_empty_state_vector()

        # Current turn
        turn = battle.turn  # Int

        # Field: Dict[Field: int=Start Turn]
        for key, value in battle.fields.items():
            state["battle"]["fields"][key.value - 1] = 1

        # Weather: Dict[Weather: int=Start Turn]
        for key, value in battle.weather.items():
            state["battle"]["weather"][key.value - 1] = 1

        # Team Info: 6 Pokemon, Team Conditions (Entry Hazards, Reflect etc.)
        # side_conditions: Dict[SideCondition: int=Start Turn/Stacks]
        for key, value in battle.side_conditions.items():
            # This is to handle the stackable conditions (spikes/toxic spikes)
            if key.value in [15, 19]:
                index = key.value + value - 1  # value = num_stacks = [1,3]
            elif key.value > 19:
                index = key.value + 1 + 2  # +1 for Toxic Spikes; +2 for Spikes
            elif key.value > 15:
                index = key.value + 2  # +2 for Spikes
            else:
                index = key.value
            state["player_team"]["side_conditions"][index - 1] = 1

        for key, value in battle.opponent_side_conditions.items():
            # This is to handle the stackable conditions (spikes/toxic spikes)
            if key.value in [15, 19]:
                index = key.value + value - 1
            elif key.value > 19:
                index = key.value + 1 + 2
            elif key.value > 15:
                index = key.value + 2
            else:
                index = key.value
            state["opponent_team"]["side_conditions"][index - 1] = 1

        # Team: Dict[str: Pokemon]
        player_team = [x for x in battle.team.values()]
        opp_team = [x for x in battle.opponent_team.values()]

        # bool
        state["player_team"]["can_dynamax"] = battle.can_dynamax
        state["opponent_team"]["can_dynamax"] = battle.opponent_can_dynamax
        # Int or None
        state["player_team"]["dynamax_turns_left"] = (
            int(battle.dynamax_turns_left or 0) / 3
        )
        state["opponent_team"]["dynamax_turns_left"] = (
            int(battle.dynamax_turns_left or 0) / 3
        )

        # Pokemon Info: 4x Moves, Last Used Move, Last Known Item, Species, Item, Ability, Possible Abilities, Types, Move PPs, Status, HP, Stats, Boosts, isFainted, isActive, Volatiles (Leech Seed, Perish Song etc.)

        all_pokemon = {
            "player_team": battle.team,
            "opponent_team": battle.opponent_team,
        }
        for key, value in all_pokemon.items():
            for i, (_, pokemon) in enumerate(value.items()):
                # Pokemon Species - One Hot Encoding(num_pokemon)
                # Visibility: Revealed
                if pokemon.species:
                    species = pokemon.species
                    species = self.lookup["cosmetic"].get(species, species)
                    species = self.lookup["pokemon"][species]
                    state[key]["pokemon"][i]["species"] = species

                # Equipped Item: One Hot Encoding(num_items)
                # Visibility: Revealed
                item = pokemon.item
                if item != None and item != "":
                    item = self.lookup["items"][item]
                    state[key]["pokemon"][i]["item"] = item

                # Ability - One Hot Encoding(num_abilities)
                # Visibility: Revealed
                ability = pokemon.ability
                if ability:
                    ability = self.lookup["abilities"][ability]
                    state[key]["pokemon"][i]["ability"] = ability

                # Possible Abilities - One Hot Encoding(3, num_abilities)
                # Visibility: Revealed
                for j, ability in enumerate(pokemon.possible_abilities):
                    ability = self.lookup["abilities"][ability]
                    state[key]["pokemon"][i]["possible_abilities"][j] = ability

                # Moves: One Hot Encoding(4, num_moves)
                # Moves PP: 4x Float[0,1]
                # Visibility: Revealed
                # TODO: Note: Dynamax Moves are not shown here.
                for j, move in enumerate(pokemon.moves.values()):
                    # Move PP
                    pp = move.current_pp / move.max_pp
                    state[key]["pokemon"][i]["moves_pp"][j] = pp
                    # Move
                    move = str(move).split(" ")[0].lower()
                    move = self.lookup["moves"][move]
                    state[key]["pokemon"][i]["moves"][j] = move

                # Pokemon Types - One Hot Encoding(2, num_types)
                # Visibility: Revealed
                state[key]["pokemon"][i]["type"][pokemon.type_1.value - 1] = 1
                if pokemon.type_2:
                    state[key]["pokemon"][i]["type"][pokemon.type_2.value - 1] = 1

                # Volatiles/Effects: One Hot Encoding(num_effects)
                # Visibility: All
                for effect, counter in pokemon.effects.items():
                    state[key]["pokemon"][i]["effects"][effect.value - 1] = 1

                # Pokemon Status Conditions - One Hot Encoding(num_status)
                # Status Counter (Only For Toxic & Sleep) - Float
                # Visibility: All
                status = pokemon.status  # Status or None
                if status:
                    state[key]["pokemon"][i]["status"][status.value - 1] = 1
                    # Max Turns for Toxic is 16 (Does 100% Damage)
                    if status.name == "TOX":
                        state[key]["pokemon"][i]["status_counter"] = (
                            pokemon.status_counter / 16
                        )
                    # Sleep is 1-3 Turns
                    elif status.name == "SLP":
                        state[key]["pokemon"][i]["status_counter"] = (
                            pokemon.status_counter / 3
                        )

                # Base Stats: float[0,1]
                # TODO: Use Level + Base Stats to compute true values??
                # Then EVs??
                # Visibility: All
                for stat, value in pokemon.base_stats.items():
                    stat = self.lookup["stats"][stat]
                    state[key]["pokemon"][i]["base_stats"][stat] = value / 255

                # Stat Boosts: float[-1,+1]
                # Visibility: All
                for stat, boost in pokemon.boosts.items():
                    stat = self.lookup["boosts"][stat]
                    state[key]["pokemon"][i]["stat_boosts"][stat] = boost / 6

                # Pokemon's Level - float[0,1]
                # Visibility: All
                state[key]["pokemon"][i]["level"] = pokemon.level / 100

                # Pokemon's Current HP Fraction - float[0,1]
                # Visibility: All
                state[key]["pokemon"][i]["health"] = pokemon.current_hp_fraction

                # Protect Counter
                # How many successive turns has it used a protect-like move?
                # At 6 it reaches the minimum 0.001% chance of success
                # https://twitter.com/0x7bdf/status/591858756733931521
                # Visibility: All
                state[key]["pokemon"][i]["protect_counter"] = (
                    pokemon.protect_counter / 6
                )

                # Pokemon is Fainted? - Boolean
                # Visibility: All
                state[key]["pokemon"][i]["fainted"] = pokemon.fainted

                # Is Active Pokemon? - Boolean
                # Visibility: All
                state[key]["pokemon"][i]["active"] = pokemon.active

                # Is it this Pokemon's first turn after being switched in?
                # This is useful for things like Fake Out or First Impression.
                # Boolean.
                # Visibility: All
                state[key]["pokemon"][i]["first_turn"] = pokemon.first_turn

                # Must Recharge this turn (Hyper Beam etc.) - Boolean
                # Visibility: All
                state[key]["pokemon"][i]["must_recharge"] = pokemon.must_recharge

                # Preparing For Attack (Dig/Bounce/etc.) - Boolean
                # Visibility: All
                # TODO: Change this back when the bug is fixed.
                # state[key]["pokemon"][i]["preparing"] = pokemon.preparing
                if pokemon.preparing:
                    state[key]["pokemon"][i]["preparing"] = True
                else:
                    state[key]["pokemon"][i]["preparing"] = False

        # Convert State Dict to State Vector for model
        state = self.state_to_machine_readable_state(state)
        # Make Mask
        self.mask = self.make_mask(battle)

        # Return State
        return torch.cat([state, self.mask], dim=-1)

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )

    def make_mask(self, battle):
        """
        We add a large negative value to the indices that
        we want to mask. That way after softmax, the value will still
        be super low.
        """
        mask = []
        for action in range(len(self.action_space)):
            action = action - 1
            if action == -1:
                mask.append(0)
            elif (
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

    def get_action_mask(self):
        return self.mask

class FullStatePlayerTesting(FullStatePlayer):
    def __init__(self, model, *args, **kwargs):
        FullStatePlayer.__init__(self, *args, **kwargs)
        self.model = model
        print(model.summary())

    def choose_move(self, battle):
        state = self.embed_battle(battle)
        predictions = self.model.predict(state)
        action = np.argmax(predictions)
        return self._action_to_move(action, battle)

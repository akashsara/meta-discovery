# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
import numpy as np
import pandas as pd
import sys
import joblib
import copy

from poke_env.environment.weather import Weather
from poke_env.environment.field import Field
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.effect import Effect
from poke_env.environment.status import Status
from poke_env.environment.pokemon_type import PokemonType
from poke_env.utils import to_id_str
import torch

sys.path.append("./")
from agents.env_player import Gen8EnvSinglePlayerFixed
from gym.spaces import Space, Box
from poke_env.player.player import Player
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class FullStatePlayer(Gen8EnvSinglePlayerFixed):
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
        # We create the lookup first since it's needed for
        # describe_embedding() which is called in super()
        super().__init__(*args, **kwargs)

    def create_empty_state_vector(self):
        pokemon = {
            "species": self.lookup["pokemon"]["unknown_pokemon"],
            "item": self.lookup["items"]["unknown_item"],
            "ability": self.lookup["abilities"]["unknown_ability"],
            "possible_abilities": [self.lookup["abilities"]["unknown_ability"]] * 4,
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
                "can_mega_evolve": False,
                "can_z_move": False,
                "pokemon": [copy.deepcopy(pokemon) for i in range(6)],
            },
            "opponent_team": {
                "side_conditions": torch.zeros(
                    (self.lookup["max_values"]["side_conditions"])
                ),
                "can_dynamax": False,
                "dynamax_turns_left": 0,
                "can_mega_evolve": False,
                "can_z_move": False,
                "pokemon": [copy.deepcopy(pokemon) for i in range(6)],
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
                torch.tensor([state["player_team"]["can_mega_evolve"]]),
                torch.tensor([state["player_team"]["can_z_move"]]),
            ]
        ).float()
        opponent_state = torch.cat(
            [
                state["opponent_team"]["side_conditions"],
                torch.tensor([state["opponent_team"]["can_dynamax"]]),
                torch.tensor([state["opponent_team"]["dynamax_turns_left"]]),
                torch.tensor([state["opponent_team"]["can_mega_evolve"]]),
                torch.tensor([state["opponent_team"]["can_z_move"]]),
            ]
        ).float()
        teams = {
            "player_team": [],
            "opponent_team": [],
        }
        active_pokemon_index = [0, 0]  # player, opponent
        for idx, (key, value) in enumerate(teams.items()):
            for i, pokemon in enumerate(state[key]["pokemon"]):
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
                teams[key].append(pokemon_state)
                # Tracks the active Pokemon
                if pokemon["active"]:
                    active_pokemon_index[idx] = i
            teams[key] = torch.cat(teams[key])
        return torch.cat(
            [
                teams["player_team"],
                player_state,
                teams["opponent_team"],
                opponent_state,
                battle_state,
                torch.tensor(active_pokemon_index),
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
                torch.tensor([state["player_team"]["can_mega_evolve"]]),
                torch.tensor([state["player_team"]["can_z_move"]]),
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

        # bool
        state["player_team"]["can_mega_evolve"] = battle.can_mega_evolve
        state["opponent_team"]["can_mega_evolve"] = battle.opponent_can_mega_evolve
        state["player_team"]["can_z_move"] = battle.can_z_move
        state["opponent_team"]["can_z_move"] = battle.opponent_can_z_move
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
        for key, pokemon_team in all_pokemon.items():
            for i, pokemon in enumerate(pokemon_team.values()):
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
                if pokemon.preparing:
                    state[key]["pokemon"][i]["preparing"] = True
                else:
                    state[key]["pokemon"][i]["preparing"] = False
        # Convert State Dict to State Vector for model
        state = self.state_to_machine_readable_state(state)
        # Make Mask
        self.mask = self.make_mask(battle)
        # Return State
        return state

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2, hp_value=1, victory_value=30
        )

    def describe_embedding(self) -> Space:
        # species, item, ability, possible_ability1-3, move1-4, type1-18
        # effects1-164, status1-7, base_stats1-6, stat_boosts1-7, moves_pp1-4
        # level, health, protect_counter, status_counter, fainted, active
        # first_turn, must_recharge, preparing
        pokemon_state_mins = (
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [0] * self.lookup["max_values"]["types"]
            + [0] * self.lookup["max_values"]["effects"]
            + [0] * self.lookup["max_values"]["status"]
            + [0] * self.lookup["max_values"]["stats"]
            + [-1] * self.lookup["max_values"]["boosts"]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        pokemon_state_maxes = (
            [self.lookup["max_values"]["species"]]
            + [self.lookup["max_values"]["items"]]
            + [self.lookup["max_values"]["abilities"]] * 5
            + [self.lookup["max_values"]["moves"]] * 4
            + [1] * self.lookup["max_values"]["types"]
            + [1] * self.lookup["max_values"]["effects"]
            + [1] * self.lookup["max_values"]["status"]
            + [1] * self.lookup["max_values"]["stats"]
            + [1] * self.lookup["max_values"]["boosts"]
            + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
        # side_conditions1-23, can_dynamax, dynamax_turns_left, can_mega_evolve
        # can_z_move
        team_state_mins = (
            pokemon_state_mins * 6
            + [0] * self.lookup["max_values"]["side_conditions"]
            + [0, 0, 0, 0]
        )
        team_state_maxes = (
            pokemon_state_maxes * 6
            + [1] * self.lookup["max_values"]["side_conditions"]
            + [1, 1, 1, 1]
        )
        # weather1-8, fields1-13
        battle_state_mins = (
            team_state_mins
            + team_state_mins
            + [0] * self.lookup["max_values"]["weather"]
            + [0] * self.lookup["max_values"]["fields"]
        )
        battle_state_maxes = (
            team_state_maxes
            + team_state_maxes
            + [1] * self.lookup["max_values"]["weather"]
            + [1] * self.lookup["max_values"]["fields"]
        )
        # player_active_pokemon_index, opponent_active_pokemon_index
        active_index_mins = [0, 0]
        active_index_maxes = [5, 5]
        full_state_mins = battle_state_mins + active_index_mins
        full_state_maxes = battle_state_maxes + active_index_maxes
        return Box(
            np.array(full_state_mins, dtype=np.float32),
            np.array(full_state_maxes, dtype=np.float32),
            dtype=np.float32,
        )

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
        # Another buggy scenario
        # https://github.com/hsahovic/poke-env/issues/320
        elif battle.opponent_active_pokemon is None:
            self.skip_step = True

        return torch.tensor(mask).float()

    def action_masks(self):
        return self.mask


class GeneralAPIFullStateAgent(Player):
    def __init__(self, model, device, lookup_filename, *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        self.model = model
        self.device = device
        self.is_actor_critic = "ActorCritic" in str(model)
        self.lookup = joblib.load(lookup_filename)
        # NOTE: This is subject to change based on the current generation.
        # Hard-coding it for now but this might cause issues later on.
        self.action_space = list(range(4 * 4 + 6))

    def create_empty_state_vector(self):
        pokemon = {
            "species": self.lookup["pokemon"]["unknown_pokemon"],
            "item": self.lookup["items"]["unknown_item"],
            "ability": self.lookup["abilities"]["unknown_ability"],
            "possible_abilities": [self.lookup["abilities"]["unknown_ability"]] * 4,
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
                "can_mega_evolve": False,
                "can_z_move": False,
                "pokemon": [copy.deepcopy(pokemon) for i in range(6)],
            },
            "opponent_team": {
                "side_conditions": torch.zeros(
                    (self.lookup["max_values"]["side_conditions"])
                ),
                "can_dynamax": False,
                "dynamax_turns_left": 0,
                "can_mega_evolve": False,
                "can_z_move": False,
                "pokemon": [copy.deepcopy(pokemon) for i in range(6)],
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
                torch.tensor([state["player_team"]["can_mega_evolve"]]),
                torch.tensor([state["player_team"]["can_z_move"]]),
            ]
        ).float()
        opponent_state = torch.cat(
            [
                state["opponent_team"]["side_conditions"],
                torch.tensor([state["opponent_team"]["can_dynamax"]]),
                torch.tensor([state["opponent_team"]["dynamax_turns_left"]]),
                torch.tensor([state["opponent_team"]["can_mega_evolve"]]),
                torch.tensor([state["opponent_team"]["can_z_move"]]),
            ]
        ).float()
        teams = {
            "player_team": [],
            "opponent_team": [],
        }
        active_pokemon_index = [0, 0]  # player, opponent
        for idx, (key, value) in enumerate(teams.items()):
            for i, pokemon in enumerate(state[key]["pokemon"]):
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
                teams[key].append(pokemon_state)
                # Tracks the active Pokemon
                if pokemon["active"]:
                    active_pokemon_index[idx] = i
            teams[key] = torch.cat(teams[key])
        return torch.cat(
            [
                teams["player_team"],
                player_state,
                teams["opponent_team"],
                opponent_state,
                battle_state,
                torch.tensor(active_pokemon_index),
            ]
        )

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

        # bool
        state["player_team"]["can_mega_evolve"] = battle.can_mega_evolve
        state["opponent_team"]["can_mega_evolve"] = battle.opponent_can_mega_evolve
        state["player_team"]["can_z_move"] = battle.can_z_move
        state["opponent_team"]["can_z_move"] = battle.opponent_can_z_move
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
        for key, pokemon_team in all_pokemon.items():
            for i, pokemon in enumerate(pokemon_team.values()):
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
                if pokemon.preparing:
                    state[key]["pokemon"][i]["preparing"] = True
                else:
                    state[key]["pokemon"][i]["preparing"] = False
        # Convert State Dict to State Vector for model
        state = self.state_to_machine_readable_state(state)
        # Make Mask
        self.mask = self.make_mask(battle)
        # Return State
        return state

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2, hp_value=1, victory_value=30
        )

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
        # Another buggy scenario
        # https://github.com/hsahovic/poke-env/issues/320
        elif battle.opponent_active_pokemon is None:
            self.skip_step = True

        return torch.tensor(mask).float()

    def action_masks(self):
        return self.mask

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
        try:
            state = self.embed_battle(battle)
        except:
            raise Exception()
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

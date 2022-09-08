import joblib
import numpy as np
import pandas as pd
import random
from poke_env.teambuilder.teambuilder import Teambuilder
from scripts.meta_discovery_utils import Epsilon

class MetaDiscoveryDatabase:
    """
    Wins: Raw number of battles won which had this pokemon in the winning team
    Picks: Number of battles this Pokemon was picked in.
    Winrate: Wins / Picks
    Pickrate: Picks / (2 * num_battles)
        Since there are 2 teams in each battle.
    """

    def __init__(self, moveset_database):
        # form_mapper converts alternate forms of Pokemon into the usual form 
        # this is specifically only for cosmetic changes
        # or scenarios where a pokemon changes form mid-battle
        self.form_mapper = {
            "basculinbluestriped": "basculin",
            "gastrodoneast": "gastrodon",
            "genesectdouse": "genesect",
            "magearnaoriginal": "magearna",
            "toxtricitylowkey": "toxtricity",
            "pikachukalos": "pikachu",
            "pikachuoriginal": "pikachu",
            "pikachusinnoh": "pikachu",
            "pikachuunova": "pikachu",
            "pikachuworld": "pikachu",
            "pikachualola": "pikachu",
            "pikachuhoenn": "pikachu",
            "pikachupartner": "pikachu",
            "eiscuenoice": "eiscue",
            "keldeoresolute": "keldeo",
            "mimikyubusted": "mimikyu",
            "zygardecomplete": "zygarde",  # Could be zygarde10 too
        }
        self.num_battles = 0
        self.wins = np.zeros(len(moveset_database), dtype=int)
        self.picks = np.zeros(len(moveset_database), dtype=int)
        self.winrates = np.zeros(len(moveset_database))
        self.pickrates = np.zeros(len(moveset_database))
        self.pokemon2key = {}
        self.key2pokemon = {}
        for i, pokemon in enumerate(moveset_database.keys()):
            self.pokemon2key[pokemon] = i
            self.key2pokemon[i] = pokemon

    def load(self, db_path):
        database = joblib.load(db_path)
        self.num_battles = database["num_battles"]
        self.wins = database["wins"]
        self.picks = database["picks"]
        self.winrates = database["winrates"]
        self.pickrates = database["pickrates"]
        self.pokemon2key = database["pokemon2key"]
        self.key2pokemon = database["key2pokemon"]

    def save(self, save_path):
        joblib.dump(
            {
                "num_battles": self.num_battles,
                "wins": self.wins,
                "picks": self.picks,
                "winrates": self.winrates,
                "pickrates": self.pickrates,
                "pokemon2key": self.pokemon2key,
                "key2pokemon": self.key2pokemon,
            },
            save_path,
        )

    def update_battle_statistics(self, wins, losses, num_battles):
        self.num_battles += num_battles
        wins = [self.form_mapper.get(win, win) for win in wins]
        losses = [self.form_mapper.get(loss, loss) for loss in losses]
        win_ids = [self.pokemon2key[win] for win in wins]
        pick_ids = [self.pokemon2key[pick] for pick in losses] + win_ids
        np.add.at(self.picks, pick_ids, 1)
        np.add.at(self.wins, win_ids, 1)
        self.winrates = np.where(self.picks != 0, self.wins / self.picks, 0.0)
        self.pickrates = self.picks / (2 * self.num_battles)


class Pokedex:
    """
    Create a Pokedex to ensure species clause.
    We essentially maintain a mapping of pokedex number: pokemon.
    Given a Pokemon we can find its pokedex number.
    We can then use this number to find all Pokemon that share the number.
    """

    def __init__(self, pokedex_json):
        pokedex = pd.read_json(pokedex_json).T
        pokedex["num"] = pokedex["num"].astype(int)
        self.pokemon2id = pokedex["num"].to_dict()
        id2pokemon = {}
        for value, key in self.pokemon2id.items():
            if key in id2pokemon:
                id2pokemon[key].append(value)
            else:
                id2pokemon[key] = [value]
        self.id2pokemon = id2pokemon

    def get_same_species(self, pokemon):
        return self.id2pokemon[self.pokemon2id[pokemon]]


class TeamBuilder(Teambuilder):
    """
    Note that IDs in Pokedex are not the same IDs as in MetaDiscoveryDatabase.
    This is because multiple distinct Pokemon (from a battle POV) can share
    the same Pokedex number.
    Pokedex is for enforcing the species clause and uses the Pokedex number.
    MetaDiscoveryDatabase just uses a unique ID for each distinct Pokemon.
    For example Landorus vs Landorus-T.
    """

    def __init__(
        self,
        epsilon: Epsilon,
        moveset_database: dict,
        all_keys: list,
        pokedex_json_path: str,
        ban_list: list,
    ):
        self.epsilon = epsilon
        self.movesets = moveset_database
        self.all_pokemon = all_keys
        self.pokedex = Pokedex(pokedex_json_path)
        self.teams = []
        self.ban_list = ban_list

    def weights2probabilities(self, weights: np.ndarray) -> np.ndarray:
        """
        np.choice requires probabilities not weights
        """
        summation = weights.sum()
        if summation:
            return weights / summation
        else:
            return (weights + 1) / weights.shape[0]

    def generate_team(self, database: MetaDiscoveryDatabase) -> str:
        # Pick 6 pokemon
        team = []
        ban_list = self.ban_list.copy()
        for i in range(6):
            # 1 - Epsilon chance of picking a team based on winrate
            if np.random.random() > self.epsilon.calculate_epsilon(database.num_battles):
                # Sample based on winrate
                probabilities = database.winrates
            # There is an epsilon chance of picking low-usage Pokemon
            else:
                # Sample based on 1 - pickrate
                probabilities = 1 - database.pickrates
            # Zero out probabilities of banned Pokemon
            ban_list_ids = [
                database.pokemon2key[pokemon]
                for pokemon in ban_list
                if pokemon in database.pokemon2key
            ]
            probabilities[ban_list_ids] = 0
            # Convert weights to probabilities
            probabilities = self.weights2probabilities(probabilities)
            # Pick a Pokemon
            pokemon = np.random.choice(a=self.all_pokemon, p=probabilities)
            # Select a random moveset for the selected Pokemon
            team.append(np.random.choice(list(self.movesets[pokemon].values())))
            # Identify all Pokemon of the same species to apply species clause
            same_species = self.pokedex.get_same_species(pokemon)
            # Add Pokemon to the temporarily defined banlist
            ban_list.extend(same_species)
        # Convert team to Showdown-usable format
        team = "\n\n".join(team)
        # Return selected team
        return team

    def generate_teams(self, database: MetaDiscoveryDatabase, num_teams: int):
        """
        Generates num_teams teams to be used for battles.
        Each call resets the previous pool of generated teams.
        """
        self.teams = []
        for i in range(num_teams):
            team = self.generate_team(database)
            team = self.join_team(self.parse_showdown_team(team))
            self.teams.append(team)

    def yield_team(self) -> str:
        """
        Returns a team randomly selected from the generated teams.
        """
        return random.choice(self.teams)
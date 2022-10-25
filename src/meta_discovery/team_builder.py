import pandas as pd
import random
from poke_env.teambuilder.teambuilder import Teambuilder
from policy import Epsilon
import numpy as np
from meta_discovery import MetaDiscoveryDatabase


def normalize(array: np.ndarray) -> np.ndarray:
    return (array - array.min()) / ((array.max() - array.min()) + 1e-12)


class Pokedex:
    """
    Create a Pokedex to ensure species clause.
    We essentially maintain a mapping of pokedex number: pokemon.
    Given a Pokemon we can find its pokedex number.
    We can then use this number to find all Pokemon that share the number.
    """

    def __init__(self, legal_pokemon):
        # Hard-Coding this as it should never change
        pokedex_json_path = "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json"
        type_chart_path = "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/typeChart.json"

        # Create type chart
        self.compute_type_chart(type_chart_path)

        # Load Pokedex and setup maps
        pokedex = pd.read_json(pokedex_json_path).T
        pokedex["num"] = pokedex["num"].astype(int)
        pokedex.drop(pokedex[pokedex.num <= 0].index, inplace=True)

        # For Species Clause
        self.compute_same_species(pokedex)

        # Setup IDs that we use to track POkemon
        self.pokemon2id = {pokemon: id_ for id_, pokemon in enumerate(legal_pokemon)}
        self.id2pokemon = {id_: pokemon for pokemon, id_ in self.pokemon2id.items()}

        # Calculate Pokemon BST
        pokedex["base_stat_total"] = pokedex["baseStats"].apply(
            lambda x: sum(x.values())
        )

        # Create mapping for id to types & list of base stat weights
        self.bst_weights = np.zeros((len(self.id2pokemon)))
        self.id2types = {}
        for id_, pokemon in self.id2pokemon.items():
            self.bst_weights[id_] = pokedex["base_stat_total"][pokemon]
            self.id2types[id_] = pokedex["types"][pokemon]

        # We normalize by 780, the highest possible legal BST as of Gen 8
        self.bst_weights = self.bst_weights / 780

        # Creates a table of n_types x n_pokemon.
        # This acts as an easy lookup for individual types.
        self.compute_pokemon_type_table()

    def compute_same_species(self, pokedex):
        # We do not use the pokedex number as our index here
        # The same no. can have distinct Pokemon from a battle perspective.
        # So we store the same species elsewhere
        temp = {}
        for pokemon, num in pokedex["num"].items():
            if num in temp:
                temp[num].append(pokemon)
            else:
                temp[num] = [pokemon]
        # TODO: This is kinda inefficent.
        self.same_species = {}
        for _, same_species in temp.items():
            for species in same_species:
                self.same_species[species] = same_species

    def compute_pokemon_type_table(self):
        self.pokemon_type_matrix = np.zeros(
            (len(self.type2type_id), len(self.pokemon2id))
        )
        for id_ in self.id2pokemon:
            for type_ in self.id2types[id_]:
                type_ = self.type2type_id[type_]
                value = 1 / len(self.id2types[id_])
                self.pokemon_type_matrix[type_][id_] = value

    def compute_type_chart(self, type_chart_path: str):
        x = pd.read_json(type_chart_path)
        self.type_id2type = x["name"].to_dict()
        self.type2type_id = {type_: idx for idx, type_ in self.type_id2type.items()}
        self.type_chart = np.zeros((len(self.type2type_id), len(self.type2type_id)))
        for _, row in x.iterrows():
            current_type = self.type2type_id[row["name"]]
            # Pokemon of immunity type take 0% damage from current type
            for immunity in row["immunes"]:
                self.type_chart[self.type2type_id[immunity]][current_type] = -2
            # Pokemon of resist type take 50% damage from current type
            for resist in row["weaknesses"]:
                self.type_chart[self.type2type_id[resist]][current_type] = -1
            # Pokemon of strength type take 200% damage from current type
            for strength in row["strengths"]:
                self.type_chart[self.type2type_id[strength]][current_type] = 1

    def find_type_weaknesses(self, types: list[str]) -> np.ndarray:
        """
        Calculate overall team "type"
        The final vector will contain the effectiveness of types
        against the team
        """
        return sum([self.type_chart[self.type2type_id[type_]] for type_ in types])

    def calculate_meta_type_weights(self, types: list[str]) -> np.ndarray:
        """
        Takes in a list of Pokemon types, ideally from the top N meta Pokemon.
        Calculates type effectiveness against the meta.
        So the highest numbers here would be the best types to use.
        """
        weaknesses = self.find_type_weaknesses(types)
        return normalize(weaknesses).reshape(-1, 1)

    def calculate_type_weights(self, types: list[str]) -> np.ndarray:
        team_weaknesses = self.find_type_weaknesses(types)
        # We assume that the enemy has Pokemon strong against us.
        # Now we find the types the enemy is weak to.
        counter_types = np.zeros(self.type_chart.shape[0])
        for id_ in self.type_id2type:
            if team_weaknesses[id_] > 0:
                effectiveness = self.type_chart[id_] * team_weaknesses[id_]
                counter_types += effectiveness
        # Normalize the values to a 0-1 range
        counter_types = normalize(counter_types)
        return counter_types.reshape(-1, 1)

    def types2text(self, weaknesses: np.ndarray):
        # Utility function to display a readable version of
        # calculated weaknesses
        for idx, ratio in enumerate(weaknesses):
            print(f"{self.type_id2type[idx]}: {ratio}")

    def get_same_species(self, pokemon: str) -> list[str]:
        return self.same_species[pokemon]


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
        all_pokemon: list,
        ban_list: list,
    ):
        self.epsilon = epsilon
        self.movesets = moveset_database
        self.all_pokemon = all_pokemon
        self.pokedex = Pokedex(legal_pokemon=all_pokemon)
        self.teams = []
        self.ban_list = ban_list
        # We use this to weigh the formula for picking Pokemon
        # Where the index refers to the number of Pokemon on the team
        self.c1 = [0, 1 / 3, 1 / 2, 2 / 3, 4 / 5, 9 / 10]
        self.c2 = [1 - x for x in self.c1]

    def weights2probabilities(self, weights: np.ndarray, mask_ids: list) -> np.ndarray:
        """
        np.choice requires probabilities not weights.
        """
        summation = weights.sum()
        if summation:
            return weights / summation
        else:
            probs = (weights + 1) / (weights.shape[0] - len(mask_ids))
            # Ensure that bans remain banned
            probs[mask_ids] = 0
            return probs

    def weights2softmax(self, weights: np.ndarray) -> np.ndarray:
        """
        Uses a masked softmax to generate probabilities.
        Subtract the max for numerical stability.
        """
        weights[weights == 0] = -1e32
        e_x = np.exp(weights - weights.max())
        return e_x / e_x.sum()

    def generate_team(self, database: MetaDiscoveryDatabase) -> str:
        team = []  # To hold the actual movesets
        team_pokemon = []  # to hold Pokemon IDs
        # Make a copy so we don't modify the true ban list
        ban_list = self.ban_list.copy()
        # Pick 6 pokemon
        for i in range(6):
            # 1 - Epsilon chance of picking a team based on winrate
            if np.random.random() > self.epsilon.calculate_epsilon(
                database.num_battles
            ):
                if len(team) == 0:
                    # When we don't have any Pokemon
                    # We use the winrates to choose the first Pokemon
                    weights = database.winrates.copy()
                else:
                    ## Calculate Type Score
                    # Get types of all pokemon in team_ids
                    types = [
                        type_
                        for pokemon in team_pokemon
                        for type_ in self.pokedex.id2types[
                            self.pokedex.pokemon2id[pokemon]
                        ]
                    ]
                    # Compute best types to counter the team's counter types
                    type_weights = self.pokedex.calculate_type_weights(types)
                    # Use type_weights to get pokemon weights
                    type_score = self.pokedex.pokemon_type_matrix * type_weights
                    # (n_pokemon, n_types) -> (n_pokemon, )
                    type_score = type_score.sum(axis=0)
                    ## Compute BST Score
                    bst_score = self.pokedex.bst_weights.copy()
                    ## Compute Popularity Score
                    popularity_matrix = database.popularity_matrix.copy()
                    popularity_matrix = normalize(popularity_matrix)
                    # Get indices from pokemon names
                    # Note that database.pokemon2key == pokedex.pokemon2id
                    team_indices = [
                        database.pokemon2key[pokemon] for pokemon in team_pokemon
                    ]
                    popularity_score = popularity_matrix[team_indices].sum(
                        axis=0
                    ) / len(team_indices)
                    ## Compute overall weights
                    term1 = self.c1[len(team)] * (bst_score + type_score)
                    term2 = self.c2[len(team)] * (popularity_score)
                    weights = term1 + term2

            # There is an epsilon chance of picking low-usage Pokemon
            else:
                # Sample based on 1 - pickrate
                weights = 1 - database.pickrates.copy()
            # Zero out probabilities of banned Pokemon
            ban_list_ids = [
                database.pokemon2key[pokemon]
                for pokemon in ban_list
                if pokemon in database.pokemon2key
            ]
            # Convert weights to probabilities
            probabilities = self.weights2probabilities(weights, mask_ids=ban_list_ids)
            # Pick a Pokemon
            pokemon = np.random.choice(a=self.all_pokemon, p=probabilities)
            # Select a random moveset for the selected Pokemon
            team.append(np.random.choice(list(self.movesets[pokemon].values())))
            team_pokemon.append(pokemon)
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

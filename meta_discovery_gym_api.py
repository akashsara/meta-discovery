import joblib
import os
import asyncio
import numpy as np
import pandas as pd
import random
import time
import torch
from threading import Thread
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.player.random_player import RandomPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from agents import simple_agent, full_state_agent
from models import simple_models, full_state_models
from scripts.meta_discovery_utils import Epsilon, LinearDecayEpsilon
from poke_env.player_configuration import PlayerConfiguration

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")


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
    ):
        self.epsilon = epsilon
        self.movesets = moveset_database
        self.all_pokemon = all_keys
        self.pokedex = Pokedex(pokedex_json_path)
        self.teams = []

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
        # 1 - Epsilon chance of picking a team based on winrate
        if np.random.random() > self.epsilon.calculate_epsilon(database.num_battles):
            # Sample based on winrate
            probabilities = database.winrates
        # There is an epsilon chance of picking low-usage Pokemon
        else:
            # Sample based on 1 - pickrate
            probabilities = 1 - database.pickrates
        # Pick 6 pokemon
        team = []
        for i in range(6):
            # Convert weights to probabilities
            probabilities = self.weights2probabilities(probabilities)
            # Pick a Pokemon
            pokemon = np.random.choice(a=self.all_pokemon, p=probabilities)
            # Select a random moveset for the selected Pokemon
            team.append(np.random.choice(self.movesets[pokemon]["movesets"]))
            # Identify all Pokemon of the same species
            same_species = [
                database.pokemon2key[pokemon]
                for pokemon in self.pokedex.get_same_species(pokemon)
                if pokemon in database.pokemon2key
            ]
            # Zero out probability of that Pokemon/other forms (species clause)
            probabilities[same_species] = 0
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



def get_action(player, state, actor_critic=False):
    if actor_critic:
        with torch.no_grad():
            predictions, _ = player.model(state.to(device))
    else:
        with torch.no_grad():
            predictions = player.model(state.to(device))
    return predictions.cpu()


async def battle_handler(player1, player2, num_challenges):
    await asyncio.gather(
        player1.agent.accept_challenges(player2.username, num_challenges),
        player2.agent.send_challenges(player1.username, num_challenges),
    )


def play_battles(player, num_battles):
    is_actor_critic = "ActorCritic" in str(player.model)
    for _ in range(num_battles):
        done = False
        state = player.reset()
        while not done:
            action_mask = player.action_masks()
            # Get action
            predictions = get_action(player, state, is_actor_critic)
            # Use policy
            action = np.argmax(predictions + action_mask)
            # Play move
            state, reward, done, _ = player.step(action)


def play_battles_wrapper(player1, player2, n_battles):
    # Make Two Threads; one per player
    t1 = Thread(target=lambda: play_battles(player1, n_battles))
    t1.start()
    t2 = Thread(target=lambda: play_battles(player2, n_battles))
    t2.start()
    # On the network side, send & accept N battles
    asyncio.run(battle_handler(player1, player2, n_battles))
    # Wait for thread completion
    t1.join()
    t2.join()

    player1.close(purge=False)
    player2.close(purge=False)


def setup_and_load_model(model, model_kwargs, model_path):
    model = model(**model_kwargs)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    moveset_db_path = "moveset_database.joblib"
    meta_discovery_db_path = "meta_discovery_database.joblib"
    # Used to enforce species clause
    pokedex_json_path = "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json"
    # Total num. battles to simulate
    num_battles_to_simulate = 100000
    # Num. battles between generating new teams
    team_generation_interval = 1000
    # Num. teams generated
    num_teams_to_generate = 2500
    # Exploration Factor - Epsilon
    # Prob. of using inverse pickrate over winrate
    epsilon_min = 0.2
    epsilon_max = 1.0
    epsilon_decay = num_battles_to_simulate / 5
    # Set random seed for reproducible results
    random_seed = 42

    # Load trained models to use for each agent
    player1_model_class = simple_models.SimpleActorCriticModel
    player1_model_path = "trained/Simple_PPO_Base_v2.1/model_1024000.pt"
    player1_model_kwargs = {
        "n_actions": 22,
        "n_obs": 10,
    }
    player2_model_class = simple_models.SimpleActorCriticModel
    player2_model_path = "trained/Simple_PPO_SelfPlay_v2.0/model_2047756.pt"
    player2_model_kwargs = {
        "n_actions": 22,
        "n_obs": 10,
    }
    player1_model = setup_and_load_model(
        player1_model_class, player1_model_kwargs, player1_model_path
    )
    player2_model = setup_and_load_model(
        player2_model_class, player2_model_kwargs, player2_model_path
    )

    # Create our battle agents
    player1_kwargs = {}
    player2_kwargs = {}
    
    player1 = simple_agent.SimpleRLPlayerTesting(
        battle_format="gen8anythinggoes",
        model=player1_model,
        start_timer_on_battle_start=True,
        player_configuration=PlayerConfiguration("Battle_Agent_1", None),
        opponent=None,
        start_challenging=False,
        **player1_kwargs,
    )

    player2 = simple_agent.SimpleRLPlayerTesting(
        battle_format="gen8anythinggoes",
        model=player2_model,
        start_timer_on_battle_start=True,
        player_configuration=PlayerConfiguration("Battle_Agent_2", None),
        opponent=None,
        start_challenging=False,
        **player2_kwargs,
    )

    # Use random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    _ = torch.manual_seed(random_seed)

    # Load moveset DB
    moveset_database = joblib.load(moveset_db_path)
    # Setup meta discovery database & load existing one if possible
    meta_discovery_database = MetaDiscoveryDatabase(moveset_database)
    if os.path.exists(meta_discovery_db_path):
        meta_discovery_database.load(meta_discovery_db_path)

    exploration_control = LinearDecayEpsilon(epsilon_max, epsilon_min, epsilon_decay)

    all_pokemon = list(meta_discovery_database.key2pokemon.values())
    team_builder = TeamBuilder(
        exploration_control, moveset_database, all_pokemon, pokedex_json_path
    )

    start_time = time.time()
    for i in range(num_battles_to_simulate // team_generation_interval):
        print(f"Epoch {i+1}")

        # Generate new teams
        print("Generating New Teams")
        start = time.time()
        team_builder.generate_teams(meta_discovery_database, num_teams_to_generate)
        end = time.time()
        print(f"Time Taken: {end - start}s")

        # Make agents use the generated teams
        player1._team = team_builder
        player2._team = team_builder

        # Play battles
        print("Battling")
        start = time.time()
        play_battles_wrapper(
            player1=player1,
            player2=player2,
            n_battles=team_generation_interval,
        )
        end = time.time()
        print(f"Battles Completed: {team_generation_interval}")
        print(f"Time Taken: {end - start}s")

        # Extract stats from the battles played
        # We get battle IDs from p1 since both players are in every battle.
        # This needs to be changed if we have multiple players
        player1_all_wins = []
        player1_all_losses = []
        player2_all_wins = []
        player2_all_losses = []
        for battle in player1.battles:
            p1_team = [
                pokemon.species for _, pokemon in player1.battles[battle].team.items()
            ]
            p2_team = [
                pokemon.species for _, pokemon in player2.battles[battle].team.items()
            ]
            if player1.battles[battle].won:
                player1_all_wins.extend(p1_team)
                player2_all_losses.extend(p2_team)
            else:
                player2_all_wins.extend(p2_team)
                player1_all_losses.extend(p1_team)

        # Reset trackers so we don't count battles twice
        player1.reset_battles()
        player2.reset_battles()

        # Update overall statistics
        meta_discovery_database.update_battle_statistics(
            player1_all_wins + player2_all_wins,
            player1_all_losses + player2_all_losses,
            team_generation_interval,
        )
    end_time = time.time()
    print(f"Simulation Time Taken: {end_time - start_time:.4f}")

    meta_discovery_database.save("meta_discovery_database.joblib")

    # Meta Discovery
    print(meta_discovery_database["num_battles"])
    print("Fin.")

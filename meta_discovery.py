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
from poke_env.player.random_player import RandomPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from agents import simple_agent, full_state_agent
from models import simple_models, full_state_models

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
        self.form_mapper = {
            "eiscuenoice": "eiscue",
            "keldeoresolute": "keldeo",
            "mimikyubusted": "mimikyu",
            "zygardecomplete": "zygarde", # Could be zygarde10 too
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
        self.num_battles = 0
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
        for win in wins:
            if win in self.form_mapper:
                win = self.form_mapper[win]
            if win not in self.pokemon2key:
                print(win)
        for loss in losses:
            if loss in self.form_mapper:
                loss = self.form_mapper[loss]
            if loss not in self.pokemon2key:
                print(loss)
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
    def __init__(self, exploration_factor, moveset_database, all_keys, pokedex_json):
        self.epsilon = exploration_factor
        self.movesets = moveset_database
        self.all_pokemon = all_keys
        self.pokedex = Pokedex(pokedex_json)

    def weights2probabilities(self, weights):
        """
        np.choice requires probabilities not weights
        """
        summation = weights.sum()
        if summation:
            return weights / summation
        else:
            return (weights + 1) / weights.shape[0]

    def generate_team(self, database):
        # 1 - Epsilon chance of picking a team based on winrate
        if np.random.random() > self.epsilon:
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
            same_species = [database.pokemon2key[pokemon] for pokemon in self.pokedex.get_same_species(pokemon) if pokemon in database.pokemon2key]
            # Zero out probability of that Pokemon/other forms (species clause)
            probabilities[same_species] = 0
        # Convert team to Showdown-usable format
        team = "\n\n".join(team)
        # Return selected team
        return team

    def generate_teams(self, database, num_teams):
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


def play_battle(player, num_battles):
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
        # Update battle statistics
        team = [pokemon.species for _, pokemon in player.current_battle.team.items()]
        if player.current_battle.won:
            player.all_wins.extend(team)
        else:
            player.all_losses.extend(team)



def play_battles(player1, player2, n_battles):
    # Setup the battle loop
    loop = asyncio.get_event_loop()
    # Make Two Threads; one per player
    t1 = Thread(target=lambda: play_battle(player1, n_battles))
    t1.start()
    t2 = Thread(target=lambda: play_battle(player2, n_battles))
    t2.start()
    # On the network side, send & accept N battles
    loop.run_until_complete(battle_handler(player1, player2, n_battles))
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
    num_battles_to_simulate = 1000
    # Prob. of using inverse pickrate over winrate
    exploration_factor = 0.2
    # Num. battles between generating new teams
    team_generation_interval = 100
    # Num. teams generated
    num_teams_to_generate = 1000
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
        opponent="placeholder",
        start_challenging=False,
        **player1_kwargs,
    )
    player2 = simple_agent.SimpleRLPlayerTesting(
        battle_format="gen8anythinggoes",
        model=player2_model,
        start_timer_on_battle_start=True,
        opponent="placeholder",
        start_challenging=False,
        **player1_kwargs,
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

    all_pokemon = list(meta_discovery_database.key2pokemon.values())
    team_builder = TeamBuilder(exploration_factor, moveset_database, all_pokemon, pokedex_json_path)

    start_time = time.time()
    for i in range(num_battles_to_simulate // team_generation_interval):
        print(f"Epoch {i+1}")
        # Reset tracked statistics for this new run
        # This is stuff we've defined so it's not a part of the API
        player1.reset_statistics()
        player2.reset_statistics()

        # Generate new teams
        print("Generating New Teams")
        team_builder.generate_teams(meta_discovery_database, num_teams_to_generate)

        # Make agents use the generated teams
        player1.agent._team = team_builder
        player2.agent._team = team_builder

        # Play battles
        print("Battling")
        play_battles(
            player1=player1,
            player2=player2,
            n_battles=team_generation_interval,
        )

        # Update statistics
        meta_discovery_database.update_battle_statistics(player1.all_wins + player2.all_wins, player1.all_losses + player2.all_losses, team_generation_interval)
    end_time = time.time()
    print(f"Simulation Time Taken: {end_time - start_time:.4f}")

    if meta_discovery_database.pickrates:
        print(meta_discovery_database)

    # Meta Discovery
    print(meta_discovery_database)
    print("Fin.")
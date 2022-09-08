import joblib
import os
import asyncio
import numpy as np
import random
import time
import torch
from threading import Thread
from poke_env.player.random_player import RandomPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from agents import simple_agent, full_state_agent
from models import simple_models, full_state_models
from scripts.meta_discovery_utils import LinearDecayEpsilon
from poke_env.player_configuration import PlayerConfiguration
from meta_discovery import TeamBuilder, MetaDiscoveryDatabase

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")

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
        player1.agent._team = team_builder
        player2.agent._team = team_builder

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

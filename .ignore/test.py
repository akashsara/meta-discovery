from agents.full_state_agent import FullStatePlayer
from agents.simple_agent import SimpleRLPlayer
from models import simple_models, full_state_models
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

config = {
    "create": True,
    "pokemon_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json",
    "moves_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/moves.json",
    "items_json": "https://raw.githubusercontent.com/akashsara/showdown-data/main/dist/data/items.json",
    "lookup_filename": "player_lookup_dicts.joblib",
}

player = FullStatePlayer(config, battle_format="gen8randombattle", log_level=50)
simple_player = SimpleRLPlayer(battle_format="gen8randombattle", log_level=50)
n_actions = len(simple_player.action_space)

state_length_dict = player.get_state_lengths()
model = full_state_models.BattleModel(n_actions, 32, 64, player.lookup["max_values"], state_length_dict)
print(model)
print("---" * 20)
simple_model = simple_models.SimpleModel(n_actions)
print(simple_model)

state = player.create_empty_state_vector()
x = player.state_to_machine_readable_state(state)
print(x.min(), x.max())

x = [x] * 32
x = torch.stack(x)
y = torch.rand(32, 10)

times = []
for i in range(10):
    start = time.time()
    out = model(x)
    end = time.time()
    times.append(end - start)
print(np.mean(times))

times = []
for i in range(10):
    start = time.time()
    out = simple_model(y)
    end = time.time()
    times.append(end - start)
print(np.mean(times))
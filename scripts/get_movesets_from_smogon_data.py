import requests
import joblib
from poke_env.data import to_id_str
import sys
sys.path.insert(0, "./")
import scripts.scraping_utils as utils

dataset_url = "https://play.pokemonshowdown.com/data/sets/gen8.json"
valid_metas = [
    "gen8anythinggoes",
    "gen8ubers",
    "gen8ou",
    "gen8uu",
    "gen8ru",
    "gen8nu",
    "gen8pu",
    "gen8zu",
    "gen8lc",
]

response = requests.get(dataset_url)
response.raise_for_status()
data = response.json()

dict_keys = list(data.keys())
for key in dict_keys:
    if key not in valid_metas:
        del data[key]

# Create our data structure
moveset_database = {}

# Iterate through each meta
for meta in valid_metas:
    # Since we potentially have different movesets in each category
    for category in ["dex", "stats"]:
        # Iterate through all Pokemon available
        for pokemon in data[meta][category]:
            for moveset_name in data[meta][category][pokemon]:
                moveset = data[meta][category][pokemon][moveset_name]
                # Hard code all LC Pokemon to level 100
                if "level" in moveset:
                    if moveset["level"] in [5, 100]:
                        moveset["level"] = 100
                # EDGE CASE HANDLER
                pokemon_id, pokemon_name = utils.pokemon_name_edge_case_handler(pokemon, moveset_name)
                # Get the moveset in the Showdown format
                moveset = utils.edge_case_handler(pokemon_id, moveset)
                moveset = utils.moveset2showdownformat(pokemon_name, moveset)
                # Append to our DB
                if pokemon_id in moveset_database:
                    """
                    Ignore a moveset if it was already there. Since we go from 
                    highest meta -> lowest meta, we should have a moveset that 
                    does better against good Pokemon. This might not be the 
                    best way to do things since this potentially biases things 
                    towards higher metas. But leaving it here for now.
                    """
                    if moveset_name in moveset_database[pokemon_id]["moveset_names"]:
                        continue
                    else:
                        moveset_database[pokemon_id]["movesets"].append(moveset)
                        moveset_database[pokemon_id]["moveset_names"].append(moveset_name)
                # Add to our DB if the Pokemon isn't there
                else:
                    moveset_database[pokemon_id] = {
                        "movesets": [moveset],
                        "moveset_names": [moveset_name]
                    }

joblib.dump(moveset_database, "moveset_database.joblib")

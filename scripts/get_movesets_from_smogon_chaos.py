"""
We use this to extract movesets for Pokemon we could not find movesets of via
the get_movesets_from_smogon.py script.
This isn't a perfect method of getting movesets. There is no guarantee that the 
movesets extracted via this script are good or even viable. 
However, since we don't have any other source of movesets, we do this.
The chaos data (for example: https://www.smogon.com/stats/2022-07/chaos/)
compiles all the statistics across all battles played in that tier.

We first load our existing moveset database and maintain a set of Pokemon
for which we already have movesets.
We then download the statistics for each individual tier and retain only 
the Pokemon for which we don't have movesets. 
We combine the statistics across tiers for these Pokemon.
We then generate a single moveset for them based on majority vote.
"""
import requests
import joblib
import sys
sys.path.insert(0, "./")
import scripts.scraping_utils as utils

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
months = ["05", "06", "07", "08"]
moveset_db_path = "moveset_database.joblib"

# Load our existing movesets so we can ignore Pokemon with existing data
moveset_database = joblib.load(moveset_db_path)

# Log No. Pokemon present in our DB already
print(f"No. of Pokemon in Moveset Database: {len(moveset_database)}")

# Download all Pokemon data across metas for the time period
print("Scraping Smogon Chaos Data.")
data = {}
total_battles = 0
for month in months:
    for meta in valid_metas:
        url = f"https://www.smogon.com/stats/2022-{month}/chaos/{meta}-0.json"
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        total_battles += int(result["info"]["number of battles"])
        result = result["data"]
        for pokemon, pokemon_info in result.items():
            pokemon_id, pokemon_name = utils.pokemon_name_edge_case_handler(pokemon, "Max Usage")
            # Ignore Pokemon for which we already have data
            if pokemon_id in moveset_database:
                continue
            # Create entry if Pokemon isn't there already
            if pokemon_id not in data:
                data[pokemon_id] = {"name": pokemon_name}
            # Scrape Moves
            if "moves" not in data[pokemon_id]:
                data[pokemon_id]["moves"] = {}
            for move, n_uses in pokemon_info["Moves"].items():
                data[pokemon_id]["moves"][move] = data[pokemon_id]["moves"].get(move, 0) + int(n_uses)
            # Scrape Abilities
            if "abilities" not in data[pokemon_id]:
                data[pokemon_id]["abilities"] = {}
            for ability, n_uses in pokemon_info["Abilities"].items():
                data[pokemon_id]["abilities"][ability] = data[pokemon_id]["abilities"].get(ability, 0) + int(n_uses)
            # Scrape Items
            if "items" not in data[pokemon_id]:
                data[pokemon_id]["items"] = {}
            for item, n_uses in pokemon_info["Items"].items():
                data[pokemon_id]["items"][item] = data[pokemon_id]["items"].get(item, 0) + int(n_uses)
            # Scrape Nature + EV Spreads
            if "spreads" not in data[pokemon_id]:
                data[pokemon_id]["spreads"] = {}
            for spread, n_uses in pokemon_info["Spreads"].items():
                data[pokemon_id]["spreads"][spread] = data[pokemon_id]["spreads"].get(spread, 0) + int(n_uses)
        print(f"{month}: {meta}")

# Save extracted raw data
joblib.dump(data, "chaos_data.joblib")

# Log No. new Pokemon to be added to the DB
print(f"No. of Pokemon not in Moveset Database: {len(data)}")

# Iterate through our data and create the max moveset
movesets = {}
for pokemon_id in data:
    x = data[pokemon_id]
    # Sort moves by frequency & take the top 4
    moves = list(dict(sorted(x["moves"].items(), key=lambda item: item[1], reverse=True)).keys())
    # Remove invalid entries
    moves = [move for move in moves if move not in ["", ":"]]
    moves = moves[:4]
    # Sort abilities by frequency & take the top 1
    abilities = list(dict(sorted(x["abilities"].items(), key=lambda item: item[1], reverse=True)).keys())
    # Remove invalid entries
    abilities = [ability for ability in abilities if ability not in ["", ":"]]
    ability = abilities[0]
    # Sort items by frequency & take the top 1
    items = list(dict(sorted(x["items"].items(), key=lambda item: item[1], reverse=True)).keys())
    # Remove invalid entries
    items = [item for item in items if item not in ["", ":", "nothing"]]
    item = items[0]
    # Sort Nature + EV Spreads by frequence & take the top 1
    spreads = list(dict(sorted(x["spreads"].items(), key=lambda item: item[1], reverse=True)).keys())
    # Remove invalid entries - EV total less than 500
    spreads = [spread for spread in spreads if sum([int(x) for x in spread.split(":")[1].split("/")]) > 500]
    spread = spreads[0]
    # Convert spread to Nature + EVs
    nature, evs = utils.spread2nature_and_evs(spread)
    # Get the actual species name of the Pokemon
    pokemon_name = x["name"]
    # Create moveset dict
    moveset = {
        "moves": moves,
        "item": item,
        "ability": ability,
        "evs": evs,
        "nature": nature,
    }
    # Convert to Showdown Format
    moveset = utils.edge_case_handler(pokemon_id, moveset)
    moveset = utils.moveset2showdownformat(pokemon_name, moveset)
    # Insert into our DB
    moveset_database[pokemon_id] = {
        "movesets": [moveset],
        "moveset_names": ["Max Usage"]
    }       

joblib.dump(moveset_database, "moveset_database.joblib")

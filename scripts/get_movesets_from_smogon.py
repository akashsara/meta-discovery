import requests
import joblib
from poke_env.data import to_id_str

stat_map = {
        "hp": "HP",
        "atk": "Atk",
        "def": "Def",
        "spa": "SpA",
        "spd": "SpD",
        "spe": "Spe"
    }

def format_evs(evs):
    str = ""
    for key, val in evs.items():
        str += f"{val} {stat_map[key]} / "
    return str[:-3]

def moveset2showdownformat(name, pokemon):
    output = f"{name}"
    if "gender" in pokemon:
        output += f" ({pokemon['gender']})"
    if "item" in pokemon:
        output += f" @ {pokemon['item']}"
    output += "\n"
    output += f"Ability: {pokemon['ability']}\n"
    if "level" in pokemon:
        output +=  f"Level: {pokemon['level']}\n"
    if "shiny" in pokemon:
        output += f"Shiny: Yes\n"
    if "evs" in pokemon:
        output += f"EVs: {format_evs(pokemon['evs'])}\n"
    output += f"{pokemon['nature']} Nature\n"
    if "ivs" in pokemon:
        output += f"IVs: {format_evs(pokemon['ivs'])}\n"
    for move in pokemon["moves"]:
        output += f"- {move}\n"
    return output[:-1]

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
            for name in data[meta][category][pokemon]:
                moveset = data[meta][category][pokemon][name]
                # We use the poke_env format to name Pokemon
                # For consistency in everything
                pokemon_id = to_id_str(pokemon)
                if "level" in moveset:
                    # hard code all LC Pokemon to level 100
                    if moveset["level"] in [5, 100]:
                        moveset["level"] = 100
                # TODO: EDGE CASE HANDLER
                # Zygarde-Complete starts as either 50% or 10%
                # If it specifies the name in the moveset name use that
                # Otherwise we set it as Zygarde
                if pokemon_id == "zygardecomplete":
                    if "10%" in name:
                        pokemon_id = "zygarde10"
                        pokemon_name = "Zygarde-10%"
                    else:
                        pokemon_id = "zygarde"
                        pokemon_name = "Zygarde"
                else:
                    pokemon_name = pokemon
                moveset = moveset2showdownformat(pokemon_name, moveset)
                # Append to our DB
                if pokemon_id in moveset_database:
                    # Ignore a moveset if it was already there
                    # Since we go from highest meta -> lowest meta
                    # We should have a moveset that does better
                    # against good Pokemon
                    # This might not be the best way to do things since 
                    # this adds some bias. But leaving it here for now.
                    if name in moveset_database[pokemon_id]["moveset_names"]:
                        continue
                    else:
                        moveset_database[pokemon_id]["movesets"].append(moveset)
                        moveset_database[pokemon_id]["moveset_names"].append(name)
                # Add to our DB if the Pokemon isn't there
                else:
                    moveset_database[pokemon_id] = {
                        "movesets": [moveset],
                        "moveset_names": [name]
                    }

joblib.dump(moveset_database, "moveset_database.joblib")

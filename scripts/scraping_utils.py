from poke_env.data import to_id_str

stat_map = {
    "hp": "HP",
    "atk": "Atk",
    "def": "Def",
    "spa": "SpA",
    "spd": "SpD",
    "spe": "Spe",
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
        output += f"Level: {pokemon['level']}\n"
    if "shiny" in pokemon:
        output += f"Shiny: Yes\n"
    if "happiness" in pokemon:
        output += f"Happiness: {pokemon['happiness']}\n"
    if "evs" in pokemon:
        output += f"EVs: {format_evs(pokemon['evs'])}\n"
    output += f"{pokemon['nature']} Nature\n"
    if "ivs" in pokemon:
        output += f"IVs: {format_evs(pokemon['ivs'])}\n"
    for move in pokemon["moves"]:
        output += f"- {move}\n"
    return output[:-1]


def spread2nature_and_evs(spread):
    """
    Converts a given Nature+EV Spread like 'Careful:0/252/0/0/4/252'
    to a string & dict for the nature and EVs like:
    Careful, {"Atk": 252, "SpD": 4, "Spe": 252,}
    """
    nature, spread = spread.split(":")
    evs = {}
    stats = ["hp", "atk", "def", "spa", "spd", "spe"]
    for stat, value in zip(stats, spread.split("/")):
        if int(value) > 0:
            evs[stat] = value
    return nature, evs


def pokemon_name_edge_case_handler(pokemon, moveset_name):
    """
    This is distinct from the normal edge case handler because it
    has to do with the Pokemon species, not any attributes.
    So this runs before we initialize our DB.
    """
    # We use the poke_env format to name Pokemon
    # For consistency in everything
    pokemon_id = to_id_str(pokemon)
    if pokemon_id == "zygardecomplete":
        """
        Zygarde-Complete starts as either 50% or 10%
        If it specifies the name in the moveset name use that
        Otherwise we set it as Zygarde
        """
        if "10%" in moveset_name:
            pokemon_id = "zygarde10"
            pokemon_name = "Zygarde-10%"
        else:
            pokemon_id = "zygarde"
            pokemon_name = "Zygarde"
    elif pokemon_id == "darmanitangalarzen":
        pokemon_id = "darmanitangalar"
        pokemon_name = "Darmanitan-Galar"
    elif pokemon_id == "darmanitanzen":
        pokemon_id = "darmanitan"
        pokemon_name = "Darmanitan"
    else:
        pokemon_name = pokemon

    return pokemon_id, pokemon_name

def edge_case_handler(pokemon_id, moveset):
    if pokemon_id in ["genesectdouse", "pichu"]: 
        # These Pokemon have to be shiny for some scenarios
        moveset["shiny"] = True
    if pokemon_id == "poliwhirl":
        # The scraped moveset is illegal so we define our own
        moveset["moves"] = ["waterfall", "earthquake", "hypnosis", "poweruppunch"]
    return moveset
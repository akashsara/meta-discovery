import joblib
import os
import sys

sys.path.append("./")
from common_utils import *
from poke_env.data import to_id_str

tier_mapper = {
    "anythinggoes": "AG",
    "ubers": "Uber",
    "ou": "OU",
    "uu": "UU",
    "ru": "RU",
    "nu": "NU",
    "pu": "PU",
    "zu": "ZU",
    "lc": "LC",
}
all_tiers = [
    # Anything Goes
    "AG",
    "(AG)",
    # Ubers
    "Uber",
    "(Uber)",
    # OU
    "OU",
    "(OU)",
    # UU
    "UUBL",
    "(UUBL)",
    "UU",
    "(UU)",
    # RU
    "RUBL",
    "(RUBL)",
    "RU",
    "(RU)",
    # NU
    "NUBL",
    "(NUBL)",
    "NU",
    "(NU)",
    # PU
    "PUBL",
    "(PUBL)",
    "PU",
    "(PU)",
    # ZU
    "ZUBL",
    "(ZUBL)",
    "ZU",
    "(ZU)",
    # LC
    "LC",
    "(LC)",
]


def legality_checker(moveset_database, tier, ban_list, exclusions=[]):
    """
    Based off: https://www.smogon.com/dex/ss/formats/{tier}/
    Where {tier} = ubers, ou, uu

    exclusions is a variable that we introduce for our experiments.
    We simply use it to exclude certain rules.
    """
    print("Checking Legality of Movesets")
    to_delete = []
    exclusions = [to_id_str(exclusion) for exclusion in exclusions]

    # Illegal Combinations (movename, legal_tiers, clause)
    illegal = [
        # Baton Pass is illegal in all competitive tiers
        ("batonpass", ["anythinggoes"], "Baton Pass Clause"),
        # Double Team & Minimize = Evasion Clause
        # Banned in all competitive tiers
        ("doubleteam", ["anythingoes"], "Evasion Clause"),
        ("minimize", ["anythingoes"], "Evasion Clause"),
        # Fissure, Guillotine, Horn Drill, Sheer COld = OHKO Clause
        # Banned in all competitive tiers
        ("fissure", ["anythingoes"], "OHKO Clause"),
        ("guillotine", ["anythingoes"], "OHKO Clause"),
        ("horndrill", ["anythingoes"], "OHKO Clause"),
        ("sheercold", ["anythingoes"], "OHKO Clause"),
        # Shadow Tag is banned in all competitive tiers
        ("shadowtag", ["anythingoes"], "Shadow Tag Clause"),
        # Moody Clause - banned below Ubers
        ("moody", ["anythingoes", "ubers"], "Moody Clause"),
        # Arena Trap is banned below Ubers
        ("arenatrap", ["anythingoes", "ubers"], "Arena Trap Clause"),
        # Power Construct is banned below Ubers
        ("powerconstruct", ["anythingoes", "ubers"], "Power Construct Clause"),
        # Light Clay is banned below OU
        ("lightclay", ["anythingoes", "ubers", "ou"], "Light Clay Clause"),
        # King's Rock is banned below OU
        ("kingsrock", ["anythingoes", "ubers", "out"], "King's Rock Clause"),
    ]

    for pokemon, movesets in moveset_database.items():
        for moveset_name, moveset in movesets.items():
            for (name, legal_tiers, clause) in illegal:
                moveset = to_id_str(moveset)
                if name in moveset and tier not in legal_tiers:
                    # Exclusion - If we want to ignore a banned entity
                    if any([exclusion in moveset for exclusion in exclusions]):
                        print(f"Exclusion: {pokemon}/{moveset_name}/{clause}")
                        continue
                    # Ban entity - queue for deletion
                    else:
                        print(f"{clause}: {pokemon}/{moveset_name}")
                        to_delete.append((pokemon, moveset_name))
                        break

    print("Deleting Illegal Movesets")
    for (pokemon, moveset) in to_delete:
        del moveset_database[pokemon][moveset]

    for pokemon in moveset_database:
        # If we run into a scenario where there are no movesets for a Pokemon
        # We simply add it to the ban list
        # Since a ban has caused us to remove this moveset
        # And there are no other legal movesets in our DB
        if len(moveset_database[pokemon]) == 0:
            ban_list.append(pokemon)
            print(f"NO MOVESETS LEFT: {pokemon}, {moveset_database[pokemon]}")

    return moveset_database, ban_list


def get_ban_list(current_tier, tier_list_path, exclusions=[]):
    if not os.path.exists(tier_list_path):
        raise Exception(
            f"We couldn't find the tier list file at {tier_list_path}. Please run meta_discovery/scripts/download_tiers.py"
        )

    current_tier = tier_mapper[current_tier]
    banned_tiers = []
    for tier in all_tiers:
        if tier == current_tier:
            break
        banned_tiers.append(tier)
    print("Banned Tiers:")
    print(banned_tiers)

    ban_list = []
    exclusions = [to_id_str(exclusion) for exclusion in exclusions]
    tier_list = joblib.load(tier_list_path)
    for pokemon, information in tier_list.items():
        if "tier" not in information:
            continue
        if pokemon in exclusions:
            print(f"Exclusion: {pokemon}")
            continue
        if information["tier"] in banned_tiers:
            ban_list.append(pokemon)
    return ban_list

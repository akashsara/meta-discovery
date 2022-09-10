import joblib
import os
import sys

sys.path.append("./")
from common_utils import *

tier_list_location = "meta_discovery/data/tier_data.joblib"
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


def legality_checker(moveset_database, tier, ban_list):
    """
    Based off: https://www.smogon.com/dex/ss/formats/{tier}/
    Where {tier} = ubers, ou, uu
    """
    print("Checking Legality of Movesets")
    to_delete = []
    for pokemon, movesets in moveset_database.items():
        for moveset_name, moveset in movesets.items():
            # Baton Pass is illegal in all tiers
            if (
                "Baton Pass" in moveset
                or "baton pass" in moveset
                or "batonpass" in moveset
            ):
                print(f"BATON PASS CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            # Double Team & Minimize = Evasion Clause = Banned in all tiers
            elif (
                "Double Team" in moveset
                or "double team" in moveset
                or "doubleteam" in moveset
            ):
                print(f"EVASION CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            elif "Minimize" in moveset or "minimize" in moveset:
                print(f"EVASION CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            # Fissure, Guillotine, Horn Drill, Sheer COld = OHKO Clause
            # Banned in all tiers
            elif "Fissure" in moveset or "fissure" in moveset:
                print(f"OHKO CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            elif "Guillotine" in moveset or "guillotine" in moveset:
                print(f"OHKO CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            elif (
                "Horn Drill" in moveset
                or "horn drill" in moveset
                or "horndrill" in moveset
            ):
                print(f"OHKO CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            elif "Sheer Cold" in moveset or "sheer cold" in moveset:
                print(f"OHKO CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            # Moody Clause - banned below Ubers
            elif tier != "ubers" and ("Moody" in moveset or "moody" in moveset):
                print(f"MOODY CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            # Arena Trap is banned below Ubers
            elif tier != "ubers" and (
                "Arena Trap" in moveset
                or "arena trap" in moveset
                or "arenatrap" in moveset
            ):
                print(f"ARENA TRAP CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            # Shadow Tag is banned in all competitive tiers
            elif (
                "Shadow Tag" in moveset
                or "shadow tag" in moveset
                or "shadowtag" in moveset
            ):
                print(f"SHADOW TAG CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            # Power Construct is banned below Ubers
            elif tier != "ubers" and (
                "Power Construct" in moveset
                or "power construct" in moveset
                or "powerconstruct" in moveset
            ):
                print(f"POWER CONSTRUCT CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            # Light Clay is banned below OU
            elif tier not in ["ubers", "ou"] and (
                "Light Clay" in moveset
                or "light clay" in moveset
                or "lightclay" in moveset
            ):
                print(f"LIGHT CLAY CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            # King's Rock is banned below OU
            elif tier not in ["ubers", "ou"] and (
                "King's Rock" in moveset
                or "Kings Rock" in moveset
                or "king's rock" in moveset
                or "kings rock" in moveset
                or "kingsrock" in moveset
                or "king'srock" in moveset
            ):
                print(f"KING'S ROCK CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))

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


def get_ban_list(current_tier):
    if not os.path.exists(tier_list_location):
        raise Exception(
            f"We couldn't find the tier list file at {tier_list_location}. Please run meta_discovery/scripts/download_tiers.py"
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
    tier_list = joblib.load(tier_list_location)
    for pokemon, information in tier_list.items():
        if "tier" not in information:
            continue
        if information["tier"] in banned_tiers:
            ban_list.append(pokemon)
    return ban_list

class Epsilon:
    def __init__(self):
        pass

    def calculate_epsilon(self):
        raise NotImplementedError()


class LinearDecayEpsilon(Epsilon):
    """
    Epsilon is our exploration factor AKA the probability of picking Pokemon
    based on 1 - pickrate instead of winrate.
    We use two variables here - epsilon_max, epsilon_min
    epsilon_max is our starting epsilon value, epsilon_min is the final value.
    Epsilon Decay is the number of calls to generate_teams() over which epsilon
    moves from epsilon_max to epsilon_min.

    Linear Decay:
        epsilon = (-(max_eps - min_eps) * iterations) / max_steps + max_eps
        epsilon = max(min_eps, epsilon)
    """

    def __init__(self, max_epsilon: float, min_epsilon: float, epsilon_decay: float):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def calculate_epsilon(self, iterations: int) -> float:
        epsilon = (
            self.max_epsilon
            - ((self.max_epsilon - self.min_epsilon) * iterations) / self.epsilon_decay
        )
        return max(self.min_epsilon, epsilon)


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


all_tiers_banlist = {
    "anythinggoes": [],
    "ubers": ["zacian", "zaciancrowned"],
    "ou": [
        "calyrexice",
        "calyrexshadow",
        "cinderace",
        "darmanitangalar",
        "dialga",
        "dracovish",
        "eternatus",
        "genesect",
        "giratina",
        "giratinaorigin",
        "groudon",
        "hooh",
        "kyogre",
        "kyurem",
        "kyuremblack",
        "kyuremwhite",
        "landorus",
        "lugia",
        "lunala",
        "magearna",
        "marshadow",
        "mewtwo",
        "naganadel",
        "necrozmadawnwings",
        "necrozmaduskmane",
        "palkia",
        "pheromosa",
        "rayquaza",
        "reshiram",
        "solgaleo",
        "spectrier",
        "urshifu",
        "xerneas",
        "yveltal",
        "zamazenta",
        "zamazentacrowned",
        "zekrom",
        "zygarde",
    ],
    "uubl": [
        "alakazam",
        "arctozolt",
        "blaziken",
        "dracozolt",
        "gengar",
        "hawlucha",
        "kommoo",
        "latias",
        "latios",
        "mienshao",
        "moltresgalar",
        "terrakion",
        "thundurus",
    ],
    "uu": [
        "barraskewda",
        "bisharp",
        "blacephalon",
        "blissey",
        "buzzwole",
        "clefable",
        "corviknight",
        "dragapult",
        "dragonite",
        "ferrothorn",
        "garchomp",
        "heatran",
        "kartana",
        "landorustherian",
        "magnezone",
        "mandibuzz",
        "melmetal",
        "mew",
        "ninetalesalola",
        "pelipper",
        "regieleki",
        "rillaboom",
        "slowbro",
        "slowkinggalar",
        "tapufini",
        "tapukoko",
        "tapulele",
        "tornadustherian",
        "toxapex",
        "tyranitar",
        "urshifurapidstrike",
        "victini",
        "volcanion",
        "volcarona",
        "weavile",
        "zapdos",
        "zapdosgalar",
        "zeraora",
    ],
}

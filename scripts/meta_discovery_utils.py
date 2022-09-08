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
            if (
                "Baton Pass" in moveset
                or "baton pass" in moveset
                or "batonpass" in moveset
            ):
                print(f"BATON PASS CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            elif tier != "ubers" and (
                "Arena Trap" in moveset
                or "arena trap" in moveset
                or "arenatrap" in moveset
            ):
                print(f"ARENA TRAP CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            elif tier != "ubers" and (
                "Shadow Tag" in moveset
                or "shadow tag" in moveset
                or "shadowtag" in moveset
            ):
                print(f"SHADOW TAG CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            elif tier != "ubers" and (
                "Power Construct" in moveset
                or "power construct" in moveset
                or "powerconstruct" in moveset
            ):
                print(f"POWER CONSTRUCT CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
            elif tier not in ["ubers", "ou"] and (
                "Light Clay" in moveset
                or "light clay" in moveset
                or "lightclay" in moveset
            ):
                print(f"LIGHT CLAY CLAUSE: {pokemon} - {moveset_name}")
                to_delete.append((pokemon, moveset_name))
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
}

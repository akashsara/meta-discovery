import joblib
import numpy as np

class MetaDiscoveryDatabase:
    """
    Wins: Raw number of battles won which had this pokemon in the winning team
    Picks: Number of battles this Pokemon was picked in.
    Winrate: Wins / Picks
    Pickrate: Picks / (2 * num_battles)
        Since there are 2 teams in each battle.
    """

    def __init__(self, moveset_database):
        # form_mapper converts alternate forms of Pokemon into the usual form 
        # this is specifically only for cosmetic changes
        # or scenarios where a pokemon changes form mid-battle
        self.form_mapper = {
            "basculinbluestriped": "basculin",
            "gastrodoneast": "gastrodon",
            "genesectdouse": "genesect",
            "magearnaoriginal": "magearna",
            "toxtricitylowkey": "toxtricity",
            "pikachukalos": "pikachu",
            "pikachuoriginal": "pikachu",
            "pikachusinnoh": "pikachu",
            "pikachuunova": "pikachu",
            "pikachuworld": "pikachu",
            "pikachualola": "pikachu",
            "pikachuhoenn": "pikachu",
            "pikachupartner": "pikachu",
            "eiscuenoice": "eiscue",
            "keldeoresolute": "keldeo",
            "mimikyubusted": "mimikyu",
            "zygardecomplete": "zygarde",  # Could be zygarde10 too
        }
        self.num_battles = 0
        self.wins = np.zeros(len(moveset_database), dtype=int)
        self.picks = np.zeros(len(moveset_database), dtype=int)
        self.winrates = np.zeros(len(moveset_database))
        self.pickrates = np.zeros(len(moveset_database))
        self.pokemon2key = {}
        self.key2pokemon = {}
        for i, pokemon in enumerate(moveset_database.keys()):
            self.pokemon2key[pokemon] = i
            self.key2pokemon[i] = pokemon
        self.popularity_matrix = np.zeros((len(moveset_database), len(moveset_database)))

    def load(self, db_path):
        database = joblib.load(db_path)
        self.num_battles = database["num_battles"]
        self.wins = database["wins"]
        self.picks = database["picks"]
        self.winrates = database["winrates"]
        self.pickrates = database["pickrates"]
        self.pokemon2key = database["pokemon2key"]
        self.key2pokemon = database["key2pokemon"]

    def save(self, save_path):
        joblib.dump(
            {
                "num_battles": self.num_battles,
                "wins": self.wins,
                "picks": self.picks,
                "winrates": self.winrates,
                "pickrates": self.pickrates,
                "pokemon2key": self.pokemon2key,
                "key2pokemon": self.key2pokemon,
            },
            save_path,
        )

    def calc_winrates_pickrates(self):
        self.winrates = np.where(self.picks != 0, self.wins / self.picks, 0.0)
        self.pickrates = self.picks / (2 * self.num_battles)

    def update_battle_statistics(self, player1, player2, num_battles):
        # Extract stats from the battles played
        # We get battle IDs from p1 since both players are in every battle.
        # This needs to be changed if we have multiple players
        all_wins = []
        all_losses = []
        for battle in player1.battles:
            player1_won = player1.battles[battle].won
            p1_team = [self.pokemon2key[self.form_mapper.get(pokemon.species, pokemon.species)] for pokemon in player1.battles[battle].team.values()]
            
            p2_team = [self.pokemon2key[self.form_mapper.get(pokemon.species, pokemon.species)] for pokemon in player2.battles[battle].team.values()]
            
            if player1_won:
                all_wins.extend(p1_team)
                all_losses.extend(p2_team)
            else:
                all_wins.extend(p2_team)
                all_losses.extend(p1_team)

            # Update the popularity matrix using the winning team
            winners = p1_team if player1_won else p2_team
            for i in range(len(winners) - 1):
                for j in range(i + 1, len(winners)):
                    self.popularity_matrix[winners[i]][winners[j]] += 1
                    self.popularity_matrix[winners[j]][winners[i]] += 1      
 
        self.num_battles += num_battles
        all_picks = all_wins + all_losses
        np.add.at(self.picks, all_picks, 1)
        np.add.at(self.wins, all_wins, 1)
        self.calc_winrates_pickrates()


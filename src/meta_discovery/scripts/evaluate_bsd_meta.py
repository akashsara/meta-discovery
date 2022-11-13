"""
This script is for comparing a BSD-meta with the current meta.
It returns a dictionary listing out the %age of the top_n pokemon in 
each tier.
"""

import joblib
import pandas as pd


if __name__ == "__main__":
    tier_data = "meta_discovery/data/tier_data.joblib"
    our_meta = f"meta_discovery/data/full_no_bans_v2/full_no_bans_v2.csv"
    top_n = 40  # No. Pokemon to consider

    our_meta = pd.read_csv(our_meta)

    meta_pokemon = our_meta["pokemon"][:top_n]
    tier_data = joblib.load(tier_data)

    results = {}

    for pokemon in meta_pokemon:
        if pokemon in tier_data:
            if "tier" in tier_data[pokemon]:
                tier = tier_data[pokemon]["tier"]
                tier = tier.replace("(", "").replace(")", "")
                results[tier] = results.get(tier, 0) + 1
            else:
                print(f"{pokemon}: No tier.")
        else:
            print(f"{pokemon}: Not found.")

    print(f"Smogon Tiers in %age of our meta:")
    print(results)
    temp = {tier: val * 100 / top_n for tier, val in results.items()}
    print(temp)
    #############################################################
    tiers = {}
    illegal = ['wyrdeer', 'kleavor', 'ursaluna', 'basculegion', 'basculegionf', 'sneasler', 'overqwil', 'enamorus', 'enamorustherian', 'missingno', 'syclar', 'syclant', 'revenankh', 'embirch', 'flarelm', 'pyroak', 'breezi', 'fidgit', 'rebble', 'tactite', 'stratagem', 'privatyke', 'arghonaut', 'nohface', 'kitsunoh', 'monohm', 'duohm', 'cyclohm', 'dorsoil', 'colossoil', 'protowatt', 'krilowatt', 'voodoll', 'voodoom', 'scratchet', 'tomohawk', 'necturine', 'necturna', 'mollux', 'cupra', 'argalis', 'aurumoth', 'brattler', 'malaconda', 'cawdet', 'cawmodore', 'volkritter', 'volkraken', 'snugglow', 'plasmanta', 'floatoy', 'caimanoe', 'naviathan', 'crucibelle', 'crucibellemega', 'pluffle', 'kerfluffle', 'pajantom', 'mumbao', 'jumbao', 'fawnifer', 'electrelk', 'caribolt', 'smogecko', 'smoguana', 'smokomodo', 'swirlpool', 'coribalis', 'snaelstrom', 'justyke', 'equilibra', 'solotl', 'astrolotl', 'miasmite', 'miasmaw', 'chromera', 'venomicon', 'venomiconepilogue', 'saharaja']
    tierlists = {}
    for pokemon in tier_data:
        if "gmax" in pokemon or "pokestar" in pokemon:
            continue
        if pokemon in illegal:
            continue
        if "tier" in tier_data[pokemon]:
            tier = tier_data[pokemon]["tier"]
            tier = tier.replace("(", "").replace(")", "")
            tiers[tier] = tiers.get(tier, 0) + 1
            tierlists[tier] = tierlists.get(tier, []) + [pokemon]

    print(f"%age of each tier in our meta:")
    for tier in results:
        print(f"{tier}: {results[tier] * 100/tiers[tier]:.2f}\t[{results[tier]}/{tiers[tier]}]")
import requests
import time
import numpy as np

def get_num_battles(url):
    response = requests.get(url)
    response.raise_for_status()
    return int(response.text.split("\n")[0].split(":")[-1].strip())

year_list = ["2022"]
month_list = ["01", "02", "03", "04", "05", "06", "07"]
meta_list = ["gen8anythinggoes", "gen8ubers", "gen8ou", "gen8uu", "gen8ru", "gen8nu", "gen8pu", "gen8zu", "gen8lc"]

results = {}
for year in year_list:
    for month in month_list:
        for meta in meta_list:
            url = f"https://www.smogon.com/stats/{year}-{month}/{meta}-0.txt"
            n_battles = get_num_battles(url)
            if meta in results:
                results[meta].append(n_battles)
            else:
                results[meta] = [n_battles]
            # Just so that we don't spam the Smogon API
            time.sleep(0.5)

# Average things out
averages = {}
for meta in meta_list:
    averages[meta] = np.mean(results[meta])

print(results)
print("---" * 30)
print(averages)
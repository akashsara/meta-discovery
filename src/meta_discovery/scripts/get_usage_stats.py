import pandas as pd
import requests
import time
import numpy as np
import os

def get_usage_data(url):
    response = requests.get(url)
    response.raise_for_status()
    text = response.text.split("\n")[5:-2]
    data = []
    for item in text:
        info = item.split("|")[1:-1]
        info = [x.strip() for x in info]
        info = {
            "rank": info[0],
            "pokemon": info[1],
            "weighted_usage_percentage": info[2],
            "raw_usage": info[3],
            "raw_usage_percentage": info[4],
            "real_usage": info[5],
            "real_usage_percentage": info[6],
        }
        #info: [rank, pokemon, usage-%, raw-usage, raw-usage-%, real, real-usage-%]
        data.append(info)
    data = pd.DataFrame(data)
    return data

year_list = ["2022"]
month_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
meta_list = ["gen8ou"]
rating_list = ["0"]
output_path = "usage_stats"

if not os.path.exists(output_path):
    os.makedirs(output_path)

results = {}
for year in year_list:
    for month in month_list:
        for meta in meta_list:
            for rating in rating_list:
                url = f"https://www.smogon.com/stats/{year}-{month}/{meta}-{rating}.txt"
                print(url)
                data = get_usage_data(url)
                filename = f"{meta}-{rating}-{year}-{month}.csv"
                data.to_csv(os.path.join(output_path, filename), index=False)
                # Just so that we don't spam the Smogon API
                time.sleep(0.5)
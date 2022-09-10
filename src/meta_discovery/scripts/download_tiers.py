import requests
import re
import ast
import joblib

if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/formats-data.ts"
    response = requests.get(url)
    response.raise_for_status()
    response = response.text
    # Get rid of the typescript code
    first_line = response.split("\n")[0]
    last_line = response[-2:]
    response = response.replace(first_line, "{").replace(last_line, "")
    response = re.sub("[\s\n\t]*(.*):", "\"\g<1>\":", response)
    response = response.replace(",\n\t}", "\n\t}").replace(",\n}", "\n}")
    response = re.sub("//.*\n", "\n", response)
    tiers = ast.literal_eval(response)
    joblib.dump(tiers, "meta_discovery/data/tier_data.joblib")
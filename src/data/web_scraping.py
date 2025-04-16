import requests
import pandas as pd
import os

def scrap_data(api_key, url, path, year, verbose=False):
    headers = {
        "x-apisports-key": api_key
    }
    if os.path.exists(path.format(year)):
        if verbose:
            print(f"The file is alrady downloaded in path: {path.format(year)}")
        return pd.read_csv(path.format(year))
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        data = data["response"]
        data = pd.json_normalize(data)
        data.to_csv(path.format(year), index=False)
        if verbose:
            print(f"file saved as csv from in {path.format(year)}")
        return data
    except Exception as e:
        print("Got this error:", e)


import requests
import pandas as pd
import os
from bs4 import BeautifulSoup

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

FLASK_APP_URL = 'http://127.0.0.1:5000/'

def scrape_table_from_flask():
    response = requests.get(FLASK_APP_URL)
    
    if response.status_code != 200:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')

    if not table:
        print("No table found in the response.")
        return None

    df = pd.read_html(str(table))[0]
    return df

if __name__ == '__main__':
    df = scrape_table_from_flask()
    if df is not None:
        print("Scraped Data:")
        df.to_csv("data/scraped_data.csv")

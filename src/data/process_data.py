import pandas as pd
from config import PathConfig, APIkey
from src.data.web_scraping import scrap_data
from sklearn.preprocessing import LabelEncoder
import pprint

def fetch_data(api_key, years=[2021, 2022, 2023], verbose=False):
    if verbose:
        print("\n\n", "="*10, f"Scrapping the data for 'years' :{years}", "="*10)
    data_frames = []

    for year in years:
        url = f"https://v3.football.api-sports.io/fixtures?league=39&season={year}"
        df = scrap_data(api_key, url, PathConfig.data_path, str(year), verbose=verbose)
        data_frames.append(df)

    data = pd.concat(data_frames)
    if verbose:
        print("All columns:", data.keys())
    return data

def drop_useless_columns(data, verbose=False):
    if verbose:
        print("\n\n", "="*10, "Removing useless columns", "="*10)
    cols_to_drop = [
        "score.extratime.home", 
        "score.extratime.away", 
        "score.penalty.home", 
        "score.penalty.away", 
        "fixture.status.extra",

        "fixture.id",           
        "fixture.referee",        
        "fixture.timezone",       
        "fixture.timestamp",      
        "fixture.venue.id",       
        "league.id",              
        "league.logo",            
        "league.flag",            
        "teams.home.id",          
        "teams.home.logo",        
        "teams.away.id",          
        "teams.away.logo"
    ]
    data.drop(columns=cols_to_drop, inplace=True)
    if verbose:
        print(f"Columns dropped:", pprint.pformat(cols_to_drop, width=2, compact=True))
    return data

def remove_nulls(data, verbose=False):
    if verbose:
        print("\n\n", "="*10, "Removing Null Values", "="*10)
        print("Original Size:", data.shape)
    data.dropna(inplace=True)
    if verbose:
        print("After removing nulls:", data.shape)
    return data

def feature_engineering(data, verbose=False):
    if verbose:
        print("\n\n", "="*10, "Feature Engineering", "="*10)
    data["fixture.date"] = pd.to_datetime(data["fixture.date"])
    data["match_day"] = data["fixture.date"].dt.day
    data["match_month"] = data["fixture.date"].dt.month
    data["match_year"] = data["fixture.date"].dt.year
    data["match_weekday"] = data["fixture.date"].dt.weekday
    data.drop(columns=["fixture.date"], inplace=True)
    if verbose:
        print("Replaced 'fixture.date' with ['match_day', 'match_month', 'match_year', 'match_weekday']")
    return data

def encode_teams(data, verbose=False):
    if verbose:
        print("\n\n", "="*10, "Encoding Team Names", "="*10)
    for col in ['teams.home.name', 'teams.away.name']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    if verbose:
        print("Encoded columns:", ['teams.home.name', 'teams.away.name'])
        print(data.keys())
    return data

def add_result_column(data, verbose=False):
    if verbose:
        print("\n\n", "="*10, "Generating 'match_result' label", "="*10)
    
    def get_result(row):
        if row['goals.home'] > row['goals.away']:
            return 'Home Win'
        elif row['goals.home'] < row['goals.away']:
            return 'Away Win'
        else:
            return 'Draw'
    
    data['match_result'] = data.apply(get_result, axis=1)
    if verbose:
        print("Created 'match_result' column")
    return data

def add_scoreline_column(data, verbose=False):
    if verbose:
        print("\n\n", "="*10, "Adding 'scoreline' column", "="*10)

    data["scoreline"] = data["goals.home"].astype(str) + "-" + data["goals.away"].astype(str)

    if verbose:
        print("'scoreline' column added.")
        print(data[["goals.home", "goals.away", "scoreline"]].head())
    
    return data

#after normalization that data become worse
# def normalize_data(data, verbose=False):
#     if verbose:
#         print("\n\n", "="*10, "Normalize the data", "="*10)
#     numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

#     if verbose:
#         print("Numeric columns to normalize:", numeric_cols.tolist())

#     scaler = StandardScaler()
#     data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

#     return data


def process_football_data(api_key, verbose=False):
    data = fetch_data(api_key, verbose=verbose)
    data = drop_useless_columns(data, verbose=verbose)
    data = remove_nulls(data, verbose=verbose)
    data = feature_engineering(data, verbose=verbose)
    data = encode_teams(data, verbose=verbose)
    data = add_result_column(data, verbose=False)
    data = add_scoreline_column(data, verbose=False)
    return data

if __name__ == "__main__":
    final_data = process_football_data(APIkey.api_key, verbose=False)  

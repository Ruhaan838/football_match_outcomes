import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from src.data.process_data import process_football_data
from config import APIkey

def compute_central_tendency(df, columns):
    print("\nCentral Tendency:")
    for col in columns:
        print(f"\n--- {col} ---")
        print(f"Mean  : {df[col].mean():.2f}")
        print(f"Median: {df[col].median():.2f}")
        print(f"Mode  : {df[col].mode().values[0]:.2f}")

def compute_spread(df, columns):
    print("\nSpread:")
    for col in columns:
        print(f"\n--- {col} ---")
        print(f"Min   : {df[col].min()}")
        print(f"Max   : {df[col].max()}")
        print(f"Range : {df[col].max() - df[col].min()}")
        print(f"Std   : {df[col].std():.2f}")
        print(f"Var   : {df[col].var():.2f}")
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        print(f"IQR   : {iqr:.2f}")

def check_normality(df, columns):
    print("\nNormality Check:")
    for col in columns:
        print(f"\n--- {col} ---")
        print(f"Skewness : {skew(df[col]):.2f}")
        print(f"Kurtosis : {kurtosis(df[col]):.2f}")
        
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=20, color="skyblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    df = process_football_data(APIkey.api_key)
    
    numeric_cols = [
        "goals.home", "goals.away", "score.halftime.home", "score.halftime.away",
        "score.fulltime.home", "score.fulltime.away", "match_day", "match_month",
        "match_year", "match_weekday", "teams.home.name", "teams.away.name"
    ]
    df['total_goals'] = df['goals.home'] + df['goals.away']

    compute_central_tendency(df, numeric_cols + ['total_goals'])
    compute_spread(df, numeric_cols + ['total_goals'])
    check_normality(df, numeric_cols + ['total_goals'])

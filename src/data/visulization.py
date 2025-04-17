import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data.process_data import process_football_data
from config import APIkey

def plot_goal_distributions(df):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df['goals.home'], bins=range(0, 8), color="blue")
    plt.title("Distribution of Home Goals")
    plt.xlabel("Goals")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    sns.histplot(df['goals.away'], bins=range(0, 8), color="red")
    plt.title("Distribution of Away Goals")
    plt.xlabel("Goals")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

def plot_total_goals(df):
    df['total_goals'] = df['goals.home'] + df['goals.away']

    plt.figure(figsize=(8, 5))
    sns.histplot(df['total_goals'], bins=range(0, 11), color='green', kde=True)
    plt.title("Total Goals per Match")
    plt.xlabel("Total Goals")
    plt.ylabel("Number of Matches")
    plt.grid(True)
    plt.show()
    return df

def add_match_result_column(df):
    def get_result(row):
        if row["goals.home"] > row["goals.away"]:
            return "Home Win"
        elif row["goals.home"] < row["goals.away"]:
            return "Away Win"
        else:
            return "Draw"
    df["match_result"] = df.apply(get_result, axis=1)
    return df

def plot_match_result_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="match_result", data=df, palette="Set2")
    plt.title("Match Result Distribution")
    plt.ylabel("Number of Matches")
    plt.show()

def plot_top_scorelines(df):
    df['scoreline'] = df['goals.home'].astype(str) + "-" + df['goals.away'].astype(str)
    plt.figure(figsize=(14, 6))
    top_scorelines = df['scoreline'].value_counts().nlargest(10)
    sns.barplot(x=top_scorelines.index, y=top_scorelines.values, palette='muted')
    plt.title("Top 10 Most Common Scorelines")
    plt.xlabel("Scoreline")
    plt.ylabel("Frequency")
    plt.show()

def plot_feature_correlation(df):
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    df = process_football_data(APIkey.api_key)
    plot_goal_distributions(df)
    df = plot_total_goals(df)
    df = add_match_result_column(df)
    plot_match_result_distribution(df)
    plot_top_scorelines(df)
    plot_feature_correlation(df)

import pandas as pd
import requests
from bs4 import BeautifulSoup as soup
from DataScraper import get_stats_table
import numpy as np


class League:
    def __init__(self, teams, hfa, results=None, fixtures=None):
        self.teams = teams
        self.hfa = hfa
        self.results = results
        self.fixtures = fixtures

    def get_league_average_attack(self):
        s = 0
        for team in self.teams:
            s += team.a
        return s / len(self.teams)

    def get_league_average_defence(self):
        s = 0
        for team in self.teams:
            s += team.b
        return s / len(self.teams)


class Fixture:
    def __init__(self, date, h, xg, xga, a):
        self.date = date
        self.h = h  # Ideally these would be instances of type Team.
        self.xg = xg
        self.xga = xga
        self.a = a  # Ideally these would be instances of type Team.

    def __str__(self):
        return f"{self.date}: {self.h} {self.xg} - {self.xga} {self.a}"


class Team:
    def __init__(self, name, short, a, b, results=None, fixtures=None):
        self.name = name
        self.short = short
        self.a = a
        self.b = b
        self.attack_rate = None
        self.defence_rate = None
        self.results = []
        if results is not None:
            self.add_results(results)
        self.fixtures = []
        if fixtures is not None:
            self.add_fixtures(fixtures)

    def set_attack_rate(self, attack_rate):
        self.attack_rate = attack_rate

    def set_defence_rate(self, defence_rate):
        self.defence_rate = defence_rate

    def add_results(self, results):
        for index, result in results.iterrows():
            self.add_result(result)

    def add_result(self, result):
        self.results.append(Fixture(np.datetime64(result['Date']), result['H'], result['xG'], result['xGA'], result['A']))

    def add_fixtures(self, fixtures):
        for index, fix in fixtures.iterrows():
            self.add_result(fix)

    def add_fixture(self, fixture):
        self.fixtures.append(Fixture(np.datetime64(fixture['Date']), fixture['H'], fixture['xG'], fixture['xGA'], fixture['A']))

    def sort_results(self, recent_first=True):
        self.results = sorted(self.results, key=lambda result: result.date, reverse=recent_first)

    def sort_fixtures(self, recent_first=True):
        self.fixtures = sorted(self.fixtures, key=lambda fixture: fixture.date, reverse=recent_first)

    def get_next_fixture(self, ordered=False):
        if not ordered:
            self.sort_fixtures()
        return self.fixtures[0] if len(self.fixtures) > 0 else None

    def __str__(self):
        return f"{self.name} ({self.short})"


if __name__ == '__main__':
    # We also want team to have fixtures and past results.
    curr_season = "https://fbref.com/en/comps/9/Premier-League-Stats"

    p = requests.get(curr_season)
    p = soup(p.text, 'html.parser')
    p = p.find(id='content')

    league_table = get_stats_table("overall", p)
    curr_teams = league_table[['Squad']].copy()
    ts = pd.Series(curr_teams['Squad'].values, index=curr_teams['Squad'])
    ts = ts.str.strip()

    short_df = pd.read_csv('short_names.csv')

    team_ratings = pd.read_csv('Team Ratings.csv')
    team_ratings = pd.DataFrame(team_ratings)

    teams = []
    for team in ts.values:
        short = short_df.loc[short_df['Team'] == team]['Short'].values[0]

        team_rating = team_ratings.loc[team_ratings['Team'] == team]
        teams.append(
            Team(team, short, team_rating['Attacking Strength'].values[0], team_rating['Defensive Strength'].values[0]))

    past_res = pd.read_csv('team_data_18_19.csv')
    for team in teams:
        print(past_res)
        print(team.name)
        team_mask = (past_res['H'] == team.name) | (past_res['A'] == team.name)
        team_res = past_res.loc[team_mask]
        team.add_results(team_res)
        team.sort_results(recent_first=True)

    print(teams[5].results[0])


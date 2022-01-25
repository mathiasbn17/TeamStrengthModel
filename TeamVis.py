import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataScraper import get_stats_table
import requests
from bs4 import BeautifulSoup as soup
from FootballStructs import Team, League


TEAM_COLORS = {'Newcastle Utd': ('w', 'black'),
               'Norwich City': ('#ffff00', 'g'),
               'Burnley': ('#99D6EA', '#6C1D45'),
               'Watford': ('#ffff00', 'black'),
               'Crystal Palace': ('b', 'r'),
               'Wolves': ('#ffa500', 'black'),
               'Leeds United': ('#Ac944D', 'w'),
               'Southampton': ('w', '#d71920'),
               'Everton': ('#003399', '#003399'),
               'Aston Villa': ('#670e36', '#95bfe5'),
               'Brentford': ('r', 'w'),
               'Tottenham': ('#132257', 'w'),
               'Leicester City': ('#fdbe11', '#003090'),
               'West Ham': ('#1bb1e7', '#7A263A'),
               'Arsenal': ('#EF0107', '#EF0107'),
               'Brighton': ('w', '#0057B8'),
               'Manchester Utd': ('#DA291C','#FBE122'),
               'Chelsea': ('#034694', '#034694'),
               'Liverpool': ('#c8102E', '#c8102E'),
               'Manchester City': ('#6CABDD', '#6CABDD')
               }


class Plot:
    def __init__(self, teams):
        self.teams = teams

    def standard(self, aspect='o'):
        if aspect == 'a':
            self.teams = sorted(self.teams, key=lambda team: team.attack_rate)
            stat = {self.teams[i].name: (self.teams[i].attack_rate, i+1) for i in range(len(self.teams))}
        elif aspect == 'd':
            self.teams = sorted(self.teams, key=lambda team: team.defence_rate, reverse=True)
            stat = {self.teams[i].name: (self.teams[i].defence_rate, i + 1) for i in range(len(self.teams))}
        else:
            self.teams = sorted(self.teams, key=lambda team: team.attack_rate - team.defence_rate)
            stat = {self.teams[i].name: (self.teams[i].attack_rate - self.teams[i].defence_rate, i + 1) for i in range(len(self.teams))}

        # Giving the plot proper size
        plt.figure(figsize=(11, 7))

        # Setting correct ticks on x-axis
        plt.xticks(np.arange(np.floor(min([val[0] for val in list(stat.values())]) * 10) / 10, np.ceil(max([val[0] for val in list(stat.values())]) * 10) / 10 + 0.1,
                             0.2 if aspect == 'o' else 0.1))

        # Making a nice grid pattern
        plt.grid(axis='both')

        # Plotting the values
        self.plot(stat, ms=16)

        plt.yticks([(i+1) for i in range(len(self.teams))], [team.short for team in self.teams])

        # Adding a bit of text
        if aspect == 'o':
            plt.title(label="Premier League Overall Team Strength Estimates", loc="left", fontsize=16, color='black')
            plt.xlabel("Predicted goal difference against average PL opponent")
        elif aspect == 'a':
            plt.title(label="Premier League Attacking Strength Estimates", loc="left", fontsize=16, color='black')
            plt.xlabel("Predicted goals scored against average PL opponent")
        elif aspect == 'd':
            plt.title(label="Premier League Defensive Strength Estimates", loc="left", fontsize=16, color='black')
            plt.xlabel("Predicted goals conceded against average PL opponent")

    def wr_plot(self):
        team_xy = {}
        for team in self.teams:
            team_xy[team.name] = (team.attack_rate, team.defence_rate)

        plt.figure(figsize=(8, 7))

        plt.grid(axis='both')

        self.plot(team_xy, ms=20)

        self.fill_diags()

        plt.title(label="Premier League Strength Estimates", loc="left", fontsize=16, color='black')
        plt.xlabel("Predicted goals scored against average PL opponent")
        plt.ylabel("Predicted goals conceded against average PL opponent")
        plt.gca().invert_yaxis()

    def fill_diags(self):
        self.teams = sorted(self.teams, key=lambda team: team.attack_rate)
        min_x = self.teams[0].attack_rate[0]
        max_x = self.teams[len(self.teams) - 1].attack_rate[0]

        self.teams = sorted(self.teams, key=lambda team: team.defence_rate)
        min_y = self.teams[0].defence_rate[0]
        max_y = self.teams[len(self.teams) - 1].defence_rate[0]

        k = -1
        A, B, C, D = (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)

        diag_length = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        diag_steps = np.linspace(0, diag_length, 9)
        #zs = np.linspace()

        for i in range(len(diag_steps - 1)):
            if i == 0:
                continue
            else:
                step_to = diag_steps[i]
                step_from = diag_steps[i - 1]


        # Med de diagonala linjerna vill vi visa målskillnad med 0.2 increments, vi har ju allt från -0.8 till
        x = np.linspace(min_x, max_x, 50)
        y = np.linspace(max_y, max_y, 50)

        plt.fill_between(x, min_y, max_y, facecolor='b', alpha=0.05) # Just let alpha increase and decrease

    def plot(self, team_xy, ms):
        # Share of circle, we want 50/50
        r1 = 0.5
        r2 = 1

        x = [0] + np.cos(np.linspace(0, 2 * np.pi * r1, 10)).tolist()
        y = [0] + np.sin(np.linspace(0, 2 * np.pi * r1, 10)).tolist()

        xy1 = list(zip(x, y))

        x = [0] + np.cos(np.linspace(2 * np.pi * r1, 2 * np.pi * r2, 10)).tolist()
        y = [0] + np.sin(np.linspace(2 * np.pi * r1, 2 * np.pi * r2, 10)).tolist()
        xy2 = list(zip(x, y))

        for point in team_xy:
            plt.plot(team_xy[point][0], team_xy[point][1], marker=xy1, ms=ms, linestyle='None', color='black',
                     markerfacecolor=TEAM_COLORS[point][0])
        for point in team_xy:
            plt.plot(team_xy[point][0], team_xy[point][1], marker=xy2, ms=ms, linestyle='None', color='black',
                     markerfacecolor=TEAM_COLORS[point][1])


def expected_goals_against_average(bbar, a, hfa):
    return bbar*a*(hfa + 1)/2


def expected_goals_conceded_against_average(b, abar, hfa):
    return b*abar*(hfa + 1)/2


def main():

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
        teams.append(Team(team, short, team_rating['Attacking Strength'].values[0], team_rating['Defensive Strength'].values[0]))

    hfa = team_ratings.HFA.unique()
    league = League(teams, hfa)

    for team in league.teams:
        team.set_attack_rate(expected_goals_against_average(league.get_league_average_defence(), team.a, league.hfa))
        team.set_defence_rate(expected_goals_conceded_against_average(team.b, league.get_league_average_attack(), league.hfa))

    fig = Plot(teams)
    #fig.standard(aspect='o')
    fig.wr_plot()
    plt.show()


if __name__ == '__main__':
    main()


"""
TO DO:
 
- Add diagonals to WR plot
- Final table predictions.
- Movement in last n GWs
"""


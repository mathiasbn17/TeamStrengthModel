import pygame as pg
import pandas as pd
import requests
from fpl import FPL
import aiohttp
import asyncio
import json
import numpy as np
from datetime import datetime

black = (40, 40, 40)
white = (255, 255, 255)
shadow = (192, 192, 192)
green = (0, 200, 0)
red = (255, 0, 0)
blue = (0, 220, 255)
yellow = (255, 255, 125)
star = (255, 255, 0)


def predicted_goals(a, b, hfa=1):
    return a*b*hfa


def translate_team_names(team):
    if team == 'Leicester':
        team = 'Leicester City'
    elif team == 'Leeds':
        team = 'Leeds United'
    elif team == 'Man City':
        team = 'Manchester City'
    elif team == 'Man Utd':
        team = 'Manchester Utd'
    elif team == 'Newcastle':
        team = 'Newcastle Utd'
    elif team == 'Norwich':
        team = 'Norwich City'
    elif team == 'Spurs':
        team = 'Tottenham'
    return team


class GW:
    def __init__(self, date, n, order, width, height):
        self.start_date = date
        self.number = n
        self.order = order

        self.text = str(self.number)
        self.font = pg.font.SysFont('Arial', 16)
        self.display_text = self.font.render(self.text, True, black, None)
        self.text_rect = self.display_text.get_rect()
        self.x, self.y = (order + 1) * width + (order + 1) * 2, 2
        self.center = (self.x + width / 2, height / 2)
        self.text_rect.center = self.center
        self.width, self.height = width, height
        self.rect = pg.Rect(self.x, self.y, self.width, self.height)

        # We want the color scheme to be 255 as
        self.color = yellow

    def shift(self, left):
        self.x -= 0.6 if left else -1*0.6
        self.center = (self.x + self.width / 2, self.height / 2)
        self.text_rect.center = self.center
        self.rect = pg.Rect(self.x, self.y, self.width, self.height)


class Fixture:
    def __init__(self, opponent, GS, GA, center, width, height, i, gw, average):
        xbase_center = center[0]
        ybase_center = center[1]

        x_center = gw.center[0]
        y_center = ybase_center
        xy = (x_center, y_center)

        self.GS = GS
        self.GA = GA

        self.goals_for = GS
        self.goals_against = GA
        self.league_average = average

        self.text = opponent
        self.font = pg.font.SysFont('Arial', 16)
        self.display_text = self.font.render(self.text, True, black, None)
        self.text_rect = self.display_text.get_rect()
        self.x, self.y = xy[0] - width / 2, xy[1] - height / 2
        self.center = xy
        self.text_rect.center = self.center
        self.width, self.height = width, height
        self.rect = pg.Rect(self.x, self.y, self.width, self.height)

        # We want the color scheme to be 255 as
        self.color = self.set_color()

    def set_color(self, steepness=4):
        """
        We want the color to be (255, 255, 255) when the goal difference is 0, so

        (i) grad = 0 when GS-GA = 0
        (ii) light blue --> dark blue = (180, 220, 255) --> (123, 159, 242) --> (66, 89, 195)
        (iii) light red --> dark red = (255, 169, 169) --> (249, 121, 121) --> (255, 73, 73)
        :return:
        """
        grad = (100 / (1 + np.exp(-1 * np.abs(self.GS - self.GA)))) - 50
        if self.GS - self.GA < 0:
            color = (255, 255 - steepness*grad, 255 - steepness*grad)
        else:
            color = (255 - steepness * grad, 255-steepness*grad, 255-grad)

        return color

    def shift(self, left):
        self.x -= 0.6 if left else -1*0.6
        self.center = (self.x + self.width / 2, self.y + self.height / 2)
        self.text_rect.center = self.center
        self.rect = pg.Rect(self.x, self.y, self.width, self.height)

    def change_aspect(self, key):
        if key == 1:
            self.GS = self.goals_for
            self.GA = self.goals_against
        elif key == 2:
            self.GS = self.goals_for
            self.GA = self.league_average
        elif key == 3:
            self.GS = self.league_average
            self.GA = self.goals_against

        self.color = self.set_color(steepness=4 if key == 1 else 6)


class Team:
    def __init__(self, name, center, width, height, a, b, short):
        self.name = name
        self.short = short
        self.fixtures = []
        self.a = a
        self.b = b

        # Setting team rect:
        self.text = self.name
        self.font = pg.font.SysFont('Arial', 16)
        self.display_text = self.font.render(self.text, True, black, None)
        self.text_rect = self.display_text.get_rect()
        self.center = center
        self.text_rect.center = self.center
        self.x, self.y = center[0] - width / 2, center[1] - height / 2
        self.width, self.height = width, height
        self.rect = pg.Rect(self.x, self.y, self.width, self.height)
        self.color = white

    def add_fixture(self, fixture):
        self.fixtures.append(fixture)


class FDR:
    def __init__(self, fixtures, team_data, gws, curr_gw, average):
        pg.init()

        self.fixtures = fixtures
        self.team_data = team_data

        self.window_width = 1096
        self.window_height = 800

        # We initiate the game display onto which we will draw our objects.
        self.display = pg.display.set_mode((self.window_width, self.window_height))
        pg.display.set_caption("Fixture Difficulty Ratings")

        # Setting initial game necessities
        self.exit = False
        self.clock = pg.time.Clock()

        # Setting initial game necessities
        self.exit = False
        self.clock = pg.time.Clock()
        self.left_mouse = False
        self.old_left_mouse = False
        self.mouse_xy = pg.mouse.get_pos()

        self.shift_right = False
        self.shift_left = False

        self.show_table = {
            1: True,
            2: False,
            3: False
        }

        # Setting up the FDR table:
        self.teams = []
        self.no_rows = len(fixtures)
        self.space_sz = 3
        self.cell_height = int((self.window_height - (self.no_rows + 1)*self.space_sz) / (self.no_rows + 1))
        self.cell_width = 120

        # Creating instances of our GWs.
        self.gws = []
        for k in range(len(gws)):
            self.gws.append(GW(gws[k], curr_gw + k, k, self.cell_width, self.cell_height))

        # Creating instances of our teams.
        i = 0
        for team in fixtures:

            team = translate_team_names(team)

            team_rating = team_data.loc[team_data['Team'] == team]
            attack = team_rating['Attacking Strength'].values[0]
            defence = team_rating['Defensive Strength'].values[0]
            short = team_rating['Short'].values[0]

            a = attack
            b = defence

            y = (i+1) * self.space_sz + self.cell_height * (i + 3/2)
            self.teams.append(Team(team, (self.cell_width / 2, y), self.cell_width, self.cell_height, a, b, short))
            i += 1

        # Normalizing team names
        team_names = list(self.fixtures.keys())

        j = 0
        while j < len(self.teams):
            wrong_name = team_names[j]
            fixtures[translate_team_names(wrong_name)] = fixtures.pop(wrong_name)
            j += 1

        # Now we must initiate fixtures
        hfa = team_data['HFA'][0]
        for team in self.teams:
            for fixture in fixtures[team.name]:
                opponent = translate_team_names(fixture['opponent'])
                opp_team = self.get_team(opponent)

                att_main = team.a
                def_main = team.b

                att_opp = opp_team.a
                def_opp = opp_team.b

                gw = self.find_gw(fixture['date'])
                if team.name == 'Arsenal' and fixture['opponent'] == 'Wolves':
                    print(gw.order + curr_gw)

                # fixture['opponent']
                team.add_fixture(Fixture(opp_team.short if fixture['home'] else opp_team.short.lower(), predicted_goals(att_main, def_opp, hfa if fixture['home'] else 1),
                                                                              predicted_goals(att_opp, def_main, hfa if not fixture['home'] else 1),
                                         team.center, self.cell_width, self.cell_height, len(team.fixtures), gw, average))

    def find_gw(self, date):
        fixture_date = datetime.strptime(date, "%Y-%m-%d")
        for gw in self.gws:
            gw_deadline = datetime.strptime(gw.start_date, "%Y-%m-%d")
            if (fixture_date - gw_deadline).days >= 0:
                max = gw
            else:
                return max
        return self.gws[len(self.gws) - 1]

    def get_team(self, name):
        for team in self.teams:
            if team.name == name:
                return team

    # Checks for exit event, and sets relevant instance variable if there was any.
    def check_exit(self, event):
        if event.type == pg.QUIT:
            self.exit = True
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self.exit = True

            elif event.key == pg.K_1:
                for team in self.teams:
                    for fixture in team.fixtures:
                        fixture.change_aspect(1)
            elif event.key == pg.K_2:
                for team in self.teams:
                    for fixture in team.fixtures:
                        fixture.change_aspect(2)
            elif event.key == pg.K_3:
                for team in self.teams:
                    for fixture in team.fixtures:
                        fixture.change_aspect(3)


    # Checks for mouse click events, and sets relevant instance variable if there was any.
    def check_mouse_click(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.left_mouse = True
        elif event.type == pg.MOUSEBUTTONUP:
            self.left_mouse = False

    def check_arrow_click(self):
        pressed = pg.key.get_pressed()
        if pressed[100]:
            self.shift_right = True
        else:
            self.shift_right = False
        if pressed[97]:
            self.shift_left = True
        else:
            self.shift_left = False

    # Handles pygame events, for instance if a key is pressed or a mouse button clicked.
    # One could say it sets the boolean values which the user can affect, which then
    # control the visual state of the GUI.
    def handle_user_events(self):
        for event in pg.event.get():
            self.check_exit(event)
            self.check_mouse_click(event)
        self.check_arrow_click()

    def display_teams(self):
        for team in self.teams:
            for fixture in team.fixtures:
                pg.draw.rect(self.display, fixture.color, fixture.rect)
                self.display.blit(fixture.display_text, fixture.text_rect)
            pg.draw.rect(self.display, team.color, team.rect)
            self.display.blit(team.display_text, team.text_rect)

    def display_gws(self):
        for gw in self.gws:
            pg.draw.rect(self.display, gw.color, gw.rect)
            self.display.blit(gw.display_text, gw.text_rect)

    def shift_fixtures(self):
        """
        Shifts the table to the left and to the right, depending on whether the keys 'a' or 'd' are pressed.
        """
        if self.shift_left and not self.shift_right:
            if self.gws[0].x > self.cell_width + 2:
                return False
            for gw in self.gws:
                gw.shift(False)
            for team in self.teams:
                for fixture in team.fixtures:
                    fixture.shift(False)
        elif self.shift_right and not self.shift_left:
            if self.gws[len(self.gws) - 1].x < self.window_width - self.cell_width:
                return False
            for gw in self.gws:
                gw.shift(True)
            for team in self.teams:
                for fixture in team.fixtures:
                    fixture.shift(True)

    def update_exterior(self):
        """
        Supervises exterior updates to the GUI.
        """
        self.display.fill(shadow)

        if self.shift_left or self.shift_right:
            self.shift_fixtures()
        self.display_teams()
        self.display_gws()

    def update_state(self):
        """
        Method supervising updating
        """
        # We update our basic game necessities.
        self.old_left_mouse = self.left_mouse
        self.mouse_xy = pg.mouse.get_pos()

        self.handle_user_events()

        self.update_exterior()

        pg.display.update()
        self.clock.tick()

    def run(self):
        """
        Main GUI function consisting of loop.
        """
        while not self.exit:
            self.update_state()

        pg.quit()
        quit()


def get_fixtures():
    """
    Collects fixtures from FPL API directly and returns them in a dict keyed by team name.
    :return: Dictionary over fixtures keyed by team name.
    """
    p = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    id_team = {}
    for team in p.json()['teams']:
        id_team[team['id']] = team['name']

    r = requests.get('https://fantasy.premierleague.com/api/fixtures?future=1')
    r = r.json()

    fixtures = {team: [] for team in id_team.values()}
    for fixture in r:
        home_team = id_team[fixture['team_h']]
        away_team = id_team[fixture['team_a']]
        date = fixture['kickoff_time'][:10]

        home_fixture = {
            'date': date,
            'opponent': away_team,
            'home': True
        }

        away_fixture = {
            'date': date,
            'opponent': home_team,
            'home': False
        }

        fixtures[home_team].append(home_fixture)
        fixtures[away_team].append(away_fixture)

    return fixtures


async def main():
    """
    Collects gameweek data (especially deadline dates) from FPL API. Uses Amos Bastian FPL package.
    :return: Upcoming gameweek deadlines, current GW number
    """
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        gameweeks = await fpl.get_gameweeks(include_live=False, return_json=True)
        gws = []
        past_gws = []
        for gw in gameweeks:
            date = gw['deadline_time'][:10]
            gw_date = datetime.strptime(date, "%Y-%m-%d")
            if (gw_date - datetime.today()).days >= 0:
                gws.append(date)  # is string variable
            else:
                past_gws.append(gw['id'])
        return gws, max(past_gws) + 1


def calculate_league_average(team_ratings):
    hfa = team_ratings['HFA'].values[0]
    average_attack = np.mean(team_ratings['Attacking Strength'])
    average_defence = np.mean(team_ratings['Defensive Strength'])

    average = np.mean([predicted_goals(average_attack, average_defence, hfa), predicted_goals(average_attack, average_defence)])

    print(average)

    return average


if __name__ == '__main__':

    team_ratings = pd.read_csv('Team Ratings.csv')
    team_ratings = pd.DataFrame(team_ratings)

    short_names = pd.read_csv('short_names.csv')

    team_ratings['Short'] = short_names['Short'].values

    fixtures = get_fixtures()

    gw_info = asyncio.run(main())
    GWs, curr_gw = gw_info[0], gw_info[1]

    average = calculate_league_average(team_ratings)

    fdr = FDR(fixtures=fixtures, team_data=team_ratings, gws=GWs, curr_gw=curr_gw, average=average)
    fdr.run()

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
import datetime
import pandas as pd

def get_text(cont):
    # Extract the text from the rows
    cont = str(cont).split('<')
    cont = [x for x in cont if x != '']
    cont = list(filter(lambda x: x[-1] != '>', cont))
    cont = [x.split('>')[1] for x in cont]
    return cont

def get_team_table(cursorObject, initialize):
    # Initialize the table
    if initialize:
        cursorObject.execute("DROP TABLE IF EXISTS teams")
        cursorObject.execute("CREATE TABLE teams (team_id INT PRIMARY KEY, franchise TEXT, from_year INT, years INT, games INT, wins INT, losses INT, win_loss_pct REAL, playoffs INT, division_champs INT, conference_champs INT, league_champs INT)")
    
    url = 'https://www.basketball-reference.com/teams/'
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    page_soup = soup(webpage, "html.parser")

    # Extract the table and the rows
    table_teams = page_soup.findAll("table", {"id": "teams_active"})
    teams_containers = table_teams[0].findAll("tr", {"class": "full_table"})

    for i in range(len(teams_containers)):
        # Extract the text from the rows
        franchise = get_text(teams_containers[i])

        if initialize:
            cursorObject.execute("INSERT INTO teams VALUES ({}, '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(i, franchise[0], franchise[2], franchise[4], franchise[5], franchise[6], franchise[7], franchise[8], franchise[9], franchise[10], franchise[11], franchise[12]))
        else:
            cursorObject.execute("UPDATE teams SET years = {}, games = {}, wins = {}, losses = {}, win_loss_pct = {}, playoffs = {}, division_champs = {}, conference_champs = {}, league_champs = {} WHERE franchise = '{}'".format(franchise[4], franchise[5], franchise[6], franchise[7], franchise[8], franchise[9], franchise[10], franchise[11], franchise[12], franchise[0]))

def get_scores_by_date(date: datetime.date):
    day = date.day
    month = date.strftime("%m")
    df = pd.DataFrame(columns=['Date', 'Home', 'Home Score', 'Away', 'Away Score', 'Outcome', 'OT', 'Home score per quarter', 'Away score per quarter'])
    url = "https://www.basketball-reference.com/boxscores/?month={}&day={}&year=2024".format(month, day)
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    page_soup = soup(webpage, "html.parser")

    containers_games = page_soup.findAll("div", {"class": "game_summary expanded nohover"})

    for i in range(len(containers_games)):
        home_team = get_text(containers_games[1].findAll("tr")[0])[1]
        away_team = get_text(containers_games[1].findAll("tr")[1])[1]
        home_score = get_text(containers_games[1].findAll("tr")[0])[3]
        away_score = get_text(containers_games[1].findAll("tr")[1])[3]
        outcome = 1 if home_score > away_score else 0
        ot_str = get_text(containers_games[1].findAll("tr")[1])[-2].split('\n')[0]
        if ot_str[-2:] != 'OT':
            ot = 0
        elif ot_str == 'OT':
            ot = 1
        elif ot_str[-2:] == 'OT':
            ot = ot_str[:-2]
        home_score_per_quarter = ', '.join(map(str, get_text(containers_games[1].findAll("tr")[3])[4:-1]))
        away_score_per_quarter = ', '.join(map(str, get_text(containers_games[1].findAll("tr")[4])[4:-1]))

        df.loc[i] = [date, home_team, home_score, away_team, away_score, outcome, ot, home_score_per_quarter, away_score_per_quarter]

    return df

def get_score_table(cursorObject, initialize, end_date=datetime.date.today()):

    start_date = datetime.date.today() - datetime.timedelta(days=7)

    # Initialize the table
    if initialize:
        assert end_date==datetime.date.today(), "When initializing the table, the end date must be today"

        cursorObject.execute("DROP TABLE IF EXISTS scores")
        cursorObject.execute("CREATE TABLE scores (game_id INTEGER PRIMARY KEY AUTOINCREMENT, date DATE, home_team TEXT, home_team_id INT, home_score INT, away_team TEXT, away_team_id INT, away_score INT, outcome INT, ot INT, home_score_per_quarter TEXT, away_score_per_quarter TEXT)")
        cursorObject.execute("DROP TABLE IF EXISTS metadata")
        cursorObject.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY AUTOINCREMENT, update_date DATE)")
        cursorObject.execute("INSERT INTO metadata (update_date) VALUES ('{}')".format(end_date))
    else:
        # Check if the table is up to date
        cursorObject.execute("SELECT * FROM metadata")
        last_update = cursorObject.fetchall()[0][1]
        if last_update == datetime.date.today():
            return
        else:
            cursorObject.execute("INSERT INTO metadata (update_date) VALUES ('{}')".format(end_date))

    # Get the scores
    delta = datetime.timedelta(days=1)
    iter_date = start_date
    while iter_date <= end_date:
        df = get_scores_by_date(iter_date)
        for i in range(len(df)):
            cursorObject.execute("INSERT INTO scores VALUES ({}, '{}', '{}', {}, {}, '{}', {}, {}, {}, {}, '{}', '{}')".format(i, df['Date'][i], df['Home'][i], 0, df['Home Score'][i], df['Away'][i], 0, df['Away Score'][i], df['Outcome'][i], df['OT'][i], df['Home score per quarter'][i], df['Away score per quarter'][i]))
        iter_date += delta  

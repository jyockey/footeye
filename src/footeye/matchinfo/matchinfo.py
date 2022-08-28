
class MatchInfo:
    # format: yyyymmdd
    match_date = None
    teams = []

    def __init__(self, home_team, away_team):
        self.home_team = home_team
        self.away_team = away_team


class MatchTeam:
    name = None
    lineup = []

    def __init__(self, name):
        self.name = name


class MatchPlayer:
    name = None
    number = None
    entry_minute = None
    exit_minute = None
    match_stats = None

    def __init__(self, name, number):
        self.name = name
        self.number = number


class MatchEvent:
    event_type = None
    minute = None

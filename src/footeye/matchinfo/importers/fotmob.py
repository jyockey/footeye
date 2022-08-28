# example url: https://www.fotmob.com/api/matchDetails?matchId=3787609

from urllib.request import urlopen
import json
from footeye.matchinfo import matchinfo


def loadData(url):
    response = urlopen(url)
    data = json.loads(response.read())

    # general/homeTeam general/awayTeam
    homeTeam = parseTeam(data, 'homeTeam')
    awayTeam = parseTeam(data, 'awayTeam')
    matchInfo = matchinfo.MatchInfo(homeTeam, awayTeam)
    return matchInfo


def parseTeam(data, teamKey):
    teamData = data['general'][teamKey]
    team = matchinfo.MatchTeam(teamData['name'])
    print(team.name)

    lineupData = None
    for val in data['content']['lineup']['lineup']:
        if val['teamId'] == teamData['id']:
            lineupData = val
            break

    if not lineupData:
        raise Exception('No lineup data found for team ' + str(teamData))

    players = []
    for row in lineupData['players']:
        for playerData in row:
            players.append(parsePlayer(playerData))

    for playerData in lineupData['bench']:
        players.append(parsePlayer(playerData))

    print('Parsed %s players' % (len(players)))
    return teamData


def parsePlayer(data):
    name = data['name']['firstName'] + ' ' + data['name']['lastName']
    return matchinfo.MatchPlayer(name, data['shirt'])


print(loadData('https://www.fotmob.com/api/matchDetails?matchId=3787609'))

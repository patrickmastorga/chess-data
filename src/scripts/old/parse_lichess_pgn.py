from parsita import *
import json

class PgnFlagParser(ParserContext, whitespace=r'[ ]*'):
    pgn_flag = '[' >> reg(r'[^" \[\]]*') & '"' >> reg(r'[^"\[\]]*') << '"]\n'



with open("lichess_db_standard_rated_2014-09.pgn", "r") as lichessData, open("games.json", "w") as outFile:
    games = list()
    gameData = dict()
    line = lichessData.readline()
    count = 1
    while line:
        # Start of pgn moves
        if line[0] == '1':
            gameData["moves"] = line[:-1]
            games.append(gameData)
            gameData = dict()
            print(f'Parsed game {count}', end='\r')
            if count == 10000:
                break
            count += 1


        if line[0] == '[':
            match PgnFlagParser.pgn_flag.parse(line):
                case Success(flag):
                    [key, value] = flag
                    gameData[key] = value
                case Failure(_):
                    print(f'Error parsing line: "{line}"')
                    
        line = lichessData.readline()
        
    json.dump(games, outFile)
            

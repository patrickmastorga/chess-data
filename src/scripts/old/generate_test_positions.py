import json
import random
from parsita import *
from old.pgntofen import PgnToFen

class PgnMovesParser(ParserContext, whitespace=r'\s*'):
    pgn_moves = reg(r'[0-9]+\.') >> repsep(reg(r'[a-hNBRQKx0-8\-\/\+#=]+') | lit('O-O') | lit('O-O-O'), opt(reg(r'[0-9]+\.')))


with open('games.json', 'r') as file:
    games = json.load(file)

with open('testpositions.txt', 'w') as out:
    i = 0
    while i < 100:
        game = random.choice(games)
        moves = game["moves"]

        match PgnMovesParser.pgn_moves.parse(moves):
            case Failure(_):
                print('Failed to parse ', moves)

            case Success(moveList):
                if (len(moveList) < 80):
                    continue
                stop = random.randint(0, len(moveList) - 1)
                fenCenverter = PgnToFen()
                fenCenverter.pgnToFen(moveList[:stop])
                out.write(fenCenverter.getFullFen() + '\n')
                i += 1

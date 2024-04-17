import chess.pgn
import random

with open("lichess_db_standard_rated_2014-09.pgn", "r") as pgnFile:
    with open("random_positions.txt", "w") as outFile:
        count = 0
        while True:
            game = chess.pgn.read_game(pgnFile)
            if game is None:
                break

            if game.headers["Termination"] != "Normal":
                continue

            finalPosition = game.end().board()

            if finalPosition.outcome() is None:
                continue

            if finalPosition.fullmove_number < 10:
                continue

            totalMoves = finalPosition.fullmove_number * 2 - (1 if finalPosition.turn == chess.WHITE else 0)
            for i in range(random.randint(4, totalMoves - 4)):
                game = game.next()
            
            outFile.write(game.board().fen() + "\n")
            count += 1
            print(count, end="\r")
            if count == 200:
                break


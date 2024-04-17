from stockfish import Stockfish

MAX_NUM_POSITIONS = 100

MAX_CENTIPAWNS = 50

fish = Stockfish(path="C:/Users/patri/Desktop/stockfish/stockfish-windows-x86-64-avx2.exe")

# Open the text file in read mode
with open('filtered-data/fens.txt', 'r') as file:
    # Open a new file to write the extracted substrings
    with open('filtered-data/even-fens.txt', 'w') as extracted_file:
        count = 0
        # Iterate through each line in the file
        for line in file:

            # Only look at positions where white is to move
            if 'w' not in line:
                continue
            
            # Evaluate position
            fish.set_fen_position(line.strip())
            evaluation = fish.get_evaluation()

            # Check if position is even enough +-50 centipawns
            if evaluation['type'] == 'cp' and abs(evaluation['value']) <= MAX_CENTIPAWNS:
                count += 1
                if count == MAX_NUM_POSITIONS:
                    break
                print(count, end='\r')

                extracted_file.write(line)

print("Done!")
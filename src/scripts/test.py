import h5py
import numpy as np

PEICES = [
    'P',
    'p',
    'N',
    'n',
    'B',
    'b',
    'R',
    'r',
    'Q',
    'q',
    'K',
    'k'
]

def vector_to_fen(vector):
    fen = ''
    for r in range(7, -1, -1):
        empty = 0
        for c in range(8):
            s = r * 8 + c
            for i in range(12):
                if vector[i * 64 + s]:
                    if empty:
                        fen += str(empty)
                        empty = 0
                    fen += PEICES[i]
                    break
            else:
                empty += 1
        
        if empty:
            fen += str(empty)
        if r:
            fen += '/'
    
    return fen

# Each char corresponds to a specific peice
PEICE_INDICES = {
    'P': 0,
    'p': 1,
    'N': 2,
    'n': 3,
    'B': 4,
    'b': 5,
    'R': 6,
    'r': 7,
    'Q': 8,
    'q': 9,
    'K': 10,
    'k': 11
}

def fen_to_vector(fen):
    # One hot encoded vector representing all of the peices on the board
    encoding = np.zeros(768, dtype=np.float32)

    # Set each entry to 1 for each peice on the board
    square = 56
    for char in fen:
        if char == '/':
            square -= 16
        elif char.isdigit():
            square += int(char)
        else:
            encoding[PEICE_INDICES[char] * 64 + square] = 1.0
            square += 1
    
    return encoding

def CP_to_WDL(x):
    # function for converting centipawn evaluation to win-draw-lose evaluation
    return 1 / (1 + np.exp(-x / 400))


arr1 = np.array([1, 2, 3, 4, 5, 6])
arr2 = np.array([6, 5, 4, 3, 2, 1])

narry = np.concatenate((arr1, arr2))

print(narry)



"""
with h5py.File('datasets/NNUE_train.hdf5', 'r') as f:
    print("START")
    i = 0
    while True:
        input()

        print(f['position'][i])
        print(vector_to_fen(f['position'][i]))
        eval = f['evaluation'][i]
        print(eval)
        print(400 * np.log(eval / (1 - eval)))
        i += 1
"""
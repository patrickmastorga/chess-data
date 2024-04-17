import numpy as np
from sklearn.model_selection import train_test_split
import h5py

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


def main():
    print('Reading .csv...')

    # Load the first CSV file
    data1 = np.genfromtxt('datasets/random_evals_filtered.csv', delimiter=',', dtype=str)
    fens1 = data1[1:, 0]
    evals1 = data1[1:, 1].astype(np.float32)

    # Load the second CSV file
    data2 = np.genfromtxt('datasets/lichess_evals_filtered.csv', delimiter=',', dtype=str)
    fens2 = data2[1:, 0]
    evals2 = data2[1:, 1].astype(np.float32)

    # Concatenate FEN strings and evaluations
    fens = np.concatenate((fens1, fens2))
    evals = np.concatenate((evals1, evals2))

    print('Preprocessing data...')

    # Convert FEN strings to vectors
    vectors = np.array([fen_to_vector(fen) for fen in fens])

    # Apply sigmoid function to evaluations
    evals_sigmoid = CP_to_WDL(evals)

    print("Splitting datasets...")

    # Split data into train and test sets
    train_vectors, test_vectors, train_evals, test_evals = train_test_split(vectors, evals_sigmoid, test_size=0.2, random_state=42)

    print('Saving train.hdf5...')

    # Save as HDF5 file
    with h5py.File('datasets/NNUE_train.hdf5', 'w') as f:
        f.create_dataset('position', data=train_vectors)
        f.create_dataset('evaluation', data=train_evals)

    print('Saving test.hdf5...')

    with h5py.File('datasets/NNUE_test.hdf5', 'w') as f:
        f.create_dataset('position', data=test_vectors)
        f.create_dataset('evaluation', data=test_evals)

    print('Done!')


if __name__ == "__main__":
    main()
import torch
import numpy as np
import os

from preprocess_hdf5 import fen_to_vector
from NNUE_train import SimpleNNUE

def WDL_to_CP(x):
    # function for converting centipawn evaluation to win-draw-lose evaluation
    return 400 * np.log(x / (1 - x))

def main():
    MODEL_NAME = 'NNUE1'
    #FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
    FEN = 'rnbqkb1r/pppppppp/5n2/8/4P1Q1/8/PPPP1PPP/RNB1KBNR'

    # Get path of file
    file_path = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.join(file_path, f'models/{MODEL_NAME}')

    # check if model exists
    if not os.path.exists(output_path):
        print('Can\'t find model!')
        return

    # Load saved weights and biases into model
    model = SimpleNNUE()
    model.load_state_dict(torch.load(os.path.join(output_path, f'{MODEL_NAME}.pth')))

    input = torch.tensor(fen_to_vector(FEN), dtype=torch.float32)

    with torch.no_grad():
        output = model(input)

    eval = WDL_to_CP(output.item())

    print(eval)


if __name__ == "__main__":
    main()
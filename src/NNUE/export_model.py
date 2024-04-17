import torch
import numpy as np
import os

from NNUE_train import SimpleNNUE

def main():
    MODEL_NAME = 'NNUE1'

    print("Saving parameters!")

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

    # Export model parameters
    binary_path = os.path.join(output_path, 'export')
    text_path = os.path.join(output_path, 'astext')
    os.makedirs(binary_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)

    for name, param in model.named_parameters():
        parameters = param.cpu().data.numpy().astype(np.float32)
        np.savetxt(os.path.join(text_path, f'{name}.txt'), parameters)

        # Write sparse linear weights in column major order
        if name == 'sparse_linear.weight':
            with open(os.path.join(binary_path, f'{name}.bin'), 'wb') as f:
                f.write(parameters.tobytes(order='F'))
        else:
            parameters.tofile(os.path.join(binary_path, f'{name}.bin'))
    
    print("Done!")


if __name__ == "__main__":
    main()
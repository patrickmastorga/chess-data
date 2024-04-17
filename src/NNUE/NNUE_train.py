import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


class ChessDataset(Dataset):
    def __init__(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            self.positions = torch.tensor(f['position'][:], dtype=torch.float32)
            self.evaluations = torch.tensor(f['evaluation'][:], dtype=torch.float32)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # need to put the tensors on the CUDA device
        return self.positions[idx], self.evaluations[idx]


class SimpleNNUE(nn.Module):
    def __init__(self):
        super(SimpleNNUE, self).__init__()
        self.sparse_linear = nn.Linear(768, 16)
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.clamp(self.sparse_linear(x), 0, 1)
        x = torch.clamp(self.linear1(x), 0, 1)
        return torch.sigmoid(self.linear2(x) / 400)


def main():
    # Input and output
    TRAIN_DATASET_PATH = 'C:/Users/patri/OneDrive/Code/chess_data/datasets/NNUE_train.hdf5'
    TEST_DATASET_PATH = 'C:/Users/patri/OneDrive/Code/chess_data/datasets/NNUE_test.hdf5'
    MODEL_NAME = 'NNUE2'
    BASE_MODEL_NAME = None

    # Training hyper-parameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001


    print(f"Training model: {MODEL_NAME}")

    # Get path of file
    file_path = os.path.dirname(os.path.abspath(__file__))
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Training on device:', torch.cuda.get_device_name())
    print('Loading datasets...')

    # Load datasets
    train_dataset = ChessDataset(TRAIN_DATASET_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = ChessDataset(TEST_DATASET_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print('Begin Training')

    # Initialize the model and move it to GPU
    model = SimpleNNUE().to(device)
    if BASE_MODEL_NAME is not None:
        input_path = os.path.join(file_path, f'models/{BASE_MODEL_NAME}/{BASE_MODEL_NAME}.pth')
        if not os.path.exists(input_path):
            print('Can\'t find base model!')
            return
        model.load_state_dict(torch.load(input_path))

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Get baseline loss
    running_loss = 0.0
    with torch.no_grad():
        for batch_num, (inputs, targets) in enumerate(test_dataloader):

            outputs = model(inputs.to(device))
            loss = criterion(outputs.squeeze(), targets.to(device))  # Squeeze to remove singleton dimension

            running_loss += loss.item() * inputs.size(0)

            print(f"BASELINE: Batch [{batch_num + 1}/{len(test_dataloader)}]", end='\r')
    print(f"{'':50}", end='\r')

    # Training loop
    train_losses = [running_loss / len(test_dataset),]
    test_losses = [running_loss / len(test_dataset),]

    for epoch in range(NUM_EPOCHS):
        # Training phase
        running_loss = 0.0
        for batch_num, (inputs, targets) in enumerate(train_dataloader):

            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs.squeeze(), targets.to(device))  # Squeeze to remove singleton dimension
            loss.backward()
            optimizer.step()

            # Clamp weights of linear1 and linear2 (for quantization later)
            if batch_num % 16 == 0:
                for name, param in model.named_parameters():
                    if name == 'linear1.weight':
                        param.data.clamp_(min=-1.9843, max=1.9843)
                    if name == 'linear2.weight':
                        param.data.clamp_(min=-127, max=127)

            running_loss += loss.item() * inputs.size(0)

            print(f"TRAINING: Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_num + 1}/{len(train_dataloader)}]", end='\r')

        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        print(f"{'':50}", end='\r')

        # Validation phase
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch_num, (inputs, targets) in enumerate(test_dataloader):

                outputs = model(inputs.to(device))
                loss = criterion(outputs.squeeze(), targets.to(device))  # Squeeze to remove singleton dimension

                running_loss += loss.item() * inputs.size(0)

                print(f"TESTING: Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_num + 1}/{len(test_dataloader)}]", end='\r')

        
        epoch_test_loss = running_loss / len(test_dataset)
        test_losses.append(epoch_test_loss)
        print(f"{'':50}\r Epoch {epoch + 1: 2}, Training Loss: {epoch_train_loss: .5f}, Testing Loss: {epoch_test_loss: .5f}")

    print('Saving model...')

    # Clamp weights of linear1 and linear2 (for quantization later)
    for name, param in model.named_parameters():
        if name == 'linear1.weight':
            param.data.clamp_(min=-1.9843, max=1.9843)
        if name == 'linear2.weight':
            param.data.clamp_(min=-127, max=127)

    # Create directory for model
    output_path = os.path.join(file_path, f'models/{MODEL_NAME}')
    os.makedirs(output_path, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(output_path, f'{MODEL_NAME}.pth'))

    # Retreive old model losses if needed
    if BASE_MODEL_NAME is not None:
        prev_train_losses, prev_test_losses = np.load(os.path.join(file_path, f'models/{BASE_MODEL_NAME}/{BASE_MODEL_NAME}-traininglosses.npy'))
        train_losses = np.concatenate((prev_train_losses, np.array(train_losses)))
        test_losses = np.concatenate((prev_test_losses, np.array(test_losses)))

    # Save training/testing loss
    np.save(os.path.join(output_path, f'{MODEL_NAME}-traininglosses.npy'), np.array([train_losses, test_losses]))
        
    # Plot the total-loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.savefig(f'{output_path}/training_and_test_losses.png')
    plt.show()

    print('Done!')


if __name__ == "__main__":
    main()
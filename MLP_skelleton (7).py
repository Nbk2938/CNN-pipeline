#TODO: all the neccessary imports
import numpy as np
import torch.nn as nn



# Define the MLP model
class MLP(nn.Module):
    def __init__(self, #TODO: add the neccessary parameters
                  ):
        ...
        #TODO: define all the input, hidden and output layers as well as activation functions

    def forward(self, #TODO: all the neccessary parameters
                 ):

        #TODO: define the forward pass
        return #TODO...

#TODO: define and fill the normalize_data function
def normalize_data():
    return ...

#TODO define and fill the MSE function
def mean_squared_error():
    return ...

#TODO fill the main training function
def train_model():

    # --------------------
    # 1) Set your paramters
    # --------------------

    # --------------------
    # 2) Create DataLoaders
    #    (not strictly necessary but it's the de facto standard)
    # --------------------


    # --------------------
    # 3) Define model, loss, optimizer
    # --------------------

    # --------------------
    # 4) Training loop over epochs
    # --------------------

    # Return the trained model and loss curves)
    return #...

#TODO: fill the function to plot a clean & noisy sample as well as the corresponding NN prediction
def plot_noisy_clean_predicted():
    return


if __name__ == "__main__":

    # (A) Load data (example: shape (N, H, W))
    noisy_data = np.load("noisy_images_small_1k.npy").astype(np.float32)
    clean_data = np.load("clean_images_small_1k.npy").astype(np.float32)

    # Convert to row-wise brightness profiles, shape (N, H)
    noisy_profiles = noisy_data.mean(axis=2)
    clean_profiles = clean_data.mean(axis=2)

    #TODO: call the normalize function

    #TODO convert to torch tensors and create the TensorDataset

    #TODO split into train and validation set (eg. with torches random_split function)

    #TODO: call your train_model function

    #TODO set the model to eval mode

    #TODO: call your noisy_clean_predicted function with a sample from the training and a sample from the testing dataset




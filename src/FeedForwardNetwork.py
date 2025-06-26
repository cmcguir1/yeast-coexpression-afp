import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    "Feedforward neural network"

    def __init__(self,inputSize,outputSize,hiddenLayers=[80,80,80],activation=nn.ReLU()):
        '''
        Arguements:
            -inputSize - number of input features
            -outputSize - number of output classes
            -hiddenLayers - a list of integers representing the number of neurons in each hidden layer

            -activation - a pyTorch activation function used on each hidden layer of the network
        '''
        super.__init__()

        self.layers = nn.Sequential()

        self.layers.add_module("Input Layer", nn.Linear(inputSize,hiddenLayers[0]))
        self.layers.add_module("Activation Function", activation)

        for i in range(len(hiddenLayers) - 1):
            self.layers.add_module("Hidden Layer" + i, nn.Linear(hiddenLayers[0],hiddenLayers[i+1]))
            self.layers.add_module("Activation Function", activation)
        
        self.layers.add_module("Output Layer", nn.Linear(hiddenLayers[-1],outputSize))
        

    def forward(self,x):
        "Feed forward input vector x through the network"

        return self.layers(x)
    
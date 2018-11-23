## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, dropout_prob):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d( 1, 32, 7)
        self.pool1 = nn.MaxPool2d(4, 4)
        #self.drop1 = nn.Dropout(dropout_prob)
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.drop2 = nn.Dropout(dropout_prob)
        
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        #self.drop3 = nn.Dropout(dropout_prob)
              
        self.conv4 = nn.Conv2d(128,256, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        #self.drop4 = nn.Dropout(dropout_prob)
         
        self.fc1 = nn.Linear(9216, 5000)   
        #self.fc1 = nn.Linear(43264, 8652)
        #self.fc1 = nn.Linear(12544, 8652)
        #self.fc1 = nn.Linear(6400, 1000)
        self.drop5 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(5000, 1024)
        #self.fc2 = nn.Linear(10000, 1000)
        self.drop6 = nn.Dropout(dropout_prob)
        
        self.fc3 = nn.Linear(1024, 136)
        #self.fc3 = nn.Linear(1000, 136)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.selu(self.conv1(x)))
        x = self.pool1(F.selu(self.conv1(x)))
        #x = self.drop1(x)
        
        x = self.pool2(F.selu(self.conv2(x)))
        #x = self.drop2(x)
        
        x = self.pool3(F.selu(self.conv3(x)))
        #x = self.drop3(x)
        
        x = self.pool4(F.selu(self.conv4(x)))
        #x = self.drop4(x)
        
        # Aplanar
        x = x.view(x.size(0), -1)
        #print(x.shape)
        
        x =  F.selu(self.fc1(x))
        x =  self.drop5(x)
        
        x = F.selu(self.fc2(x))
        x = self.drop6(x)
        
        x = F.selu(self.fc3(x))
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

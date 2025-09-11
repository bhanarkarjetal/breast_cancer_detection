import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimpleCNN(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int] = (3, 64, 64),
                 conv1_kernel: Tuple[int, int] = (3, 3),
                 conv1_neurons: int = 6,
                 conv2_neurons: int = 16,
                 conv2_kernel: Tuple[int, int] = (5, 5),
                 fc1_neurons: int = 120,
                 fc2_neurons: int = 84,
                 dropout_prob: float = 0.5,
                 num_classes: int = 10):

        super(SimpleCNN, self).__init__()

        # First Convolutional block
        self.conv1 = nn.Conv2d(input_size[0], 
                               out_channels=conv1_neurons, 
                               kernel_size=conv1_kernel, 
                               stride=1)
        self.bn1 = nn.BatchNorm2d(conv1_neurons)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional block
        self.conv2 = nn.Conv2d(in_channels=conv1_neurons, 
                               out_channels=conv2_neurons, 
                               kernel_size=conv2_kernel, 
                               stride=1)
        self.bn2 = nn.BatchNorm2d(conv2_neurons)

        # input dimensions for the first fully connected layer
        self.fc1_input_size = self.get_conv_output_size(input_size)

        # First Fully Connected layer 
        self.fc1 = nn.Linear(in_features=self.fc1_input_size, out_features=fc1_neurons)
        self.bn3 = nn.BatchNorm1d(fc1_neurons)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        # Second Fully Connected layer
        self.fc2 = nn.Linear(in_features=fc1_neurons, out_features=fc2_neurons)
        self.bn4 = nn.BatchNorm1d(fc2_neurons)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        # Output Layer
        self.fc3 = nn.Linear(in_features=fc2_neurons, out_features=num_classes)

    def get_conv_output_size(self, input_size):
        with torch.no_grad():
            # dummy input tensor with the given input size
            x = torch.rand(1, *input_size)
            
            # Pass through convolutional and pooling layers
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))

        # Return the number of elements in the output tensor
        return x.numel()

    def forward(self, x):
        # First convolutional block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second convolutional block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # First fully connected block: Linear -> BatchNorm -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second fully connected block: Linear -> BatchNorm -> ReLU -> Dropout
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer (raw logits)
        x = self.fc3(x)
        
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    The architecture is configurable via a dictionary.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the model based on a configuration dictionary.
        
        Args:
            config (Dict[str, Any]): A dictionary containing model hyperparameters.
                                     Expected keys:
                                     - 'input_size': Tuple[int, int, int] (C, H, W)
                                     - 'num_classes': int
                                     - 'conv_layers': List[Dict] with 'out_channels', 'kernel_size', etc.
                                     - 'fc_layers': List[Dict] with 'out_features', 'dropout_prob', etc.
        """
        
        super(SimpleCNN, self).__init__()

        # Build the convolutional feature extractor
        self.config = config
        conv_layers = []
        in_channels = self.config['input_size'][0] 

        for layer_cfg in self.config['conv_layers']:
            conv_layers.append(nn.Conv2d(in_channels=in_channels, **layer_cfg))
            conv_layers.append(nn.BatchNorm2d(layer_cfg['out_channels']))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = layer_cfg['out_channels']

        self.conv_layers = nn.Sequential(*conv_layers)

     
        # Calculate input dimensions for the first fully connected layer
        self.fc1_input_size = self._get_conv_output_size()

        # Build the fully connected classifier
        fc_layers = []
        in_features = self.fc1_input_size

        for fc_layer_config in self.config['fc_layers']:
            out_features = fc_layer_config['out_features']
            dropout_prob = fc_layer_config.get('dropout_prob', 0.5)

            fc_layers.append(nn.Linear(in_features, out_features=out_features))
            fc_layers.append(nn.BatchNorm1d(out_features))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout_prob))
            in_features = out_features

        # Add the final output layer
        fc_layers.append(nn.Linear(in_features, out_features=self.config['num_classes']))

        self.classifier = nn.Sequential(*fc_layers)
               

    def _get_conv_output_size(self) -> int:
        """
        Helper function to calculate the output size of the convolutional layers.
        """
        with torch.no_grad():
            # dummy input tensor with the given input size
            dummy_input = torch.rand(1, *self.config['input_size'])

            output = self.conv_layers(dummy_input)

        # Return the number of elements in the output tensor
        return output.numel()

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
        
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    # Example configuration
    config = {
        'input_size': (3, 64, 64),  # C, H, W
        'num_classes': 10,
        'conv_layers': [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ],
        'fc_layers': [
            {'out_features': 128, 'dropout_prob': 0.5},
            {'out_features': 64, 'dropout_prob': 0.5}
        ]
    }

    model = SimpleCNN(config)
    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 64, 64)  # Batch size of 1
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (1, num_classes)
        
    
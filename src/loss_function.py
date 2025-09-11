import torch.nn as nn

class LossFunction:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, predictions, targets):
        return self.criterion(predictions, targets)
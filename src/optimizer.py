import torch.optim as optim

class Optimizer:
    def __init__(self, model, learning_rate: float = 0.001):
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


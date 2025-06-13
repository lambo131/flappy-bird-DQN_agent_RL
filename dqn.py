import torch
from torch import nn

# // nn.functional has functions like activation functions (ReLU, sigmoid)
import torch.nn.functional as F

# //inherit parent class "nn.Module"
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, hidden_layers=2):
        super(DQN, self).__init__()
        
        # Dynamically create hidden layers
        layers = []
        # Input layer (state_dim → hidden_dim)
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Hidden layers (hidden_dim → hidden_dim)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer (hidden_dim → action_dim)
        layers.append(nn.Linear(hidden_dim, action_dim))
        # Combine all layers into a sequential model
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    

if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    # // randn(10, state_dim) generates a batch containing "10" state value set
    state = torch.randn(5, state_dim) 
    output = net(state)
    print(output)
import torch
from torch import nn

# // nn.functional has functions like activation functions (ReLU, sigmoid)
import torch.nn.functional as F

# //inherit parent class "nn.Module"
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()

        '''  
            in pytorch, the input layer is implicit, no need to define the input layer
            the first layer passed to the sequential structure will be known as the input layer

            fc1, fc2, .. means "full connected"

            a "linear" layer refers to a layer that performs a linear transformation on the input data
            output=input×weights+bias


        '''
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x): 
        # // this is a sequenctial definition of the layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    # // randn(10, state_dim) generates a batch containing "10" state value set
    state = torch.randn(5, state_dim) 
    output = net(state)
    print(output)
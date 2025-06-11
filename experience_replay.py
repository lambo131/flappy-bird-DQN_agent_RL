from collections import deque
import random
import torch
import numpy as np

class ReplayMemory():
    def __init__(self, max_len, seed=None):
        # // create a memory structure that is a queue of len=maxlen. The queue is first in first out
        self.memory = deque([],maxlen=max_len)
        if seed is not None:
            random.seed(seed)
    
    def clear(self, percentage_to_clear=1):
        if percentage_to_clear >= 1:
            self.memory.clear()
        else:
            num_to_clear = int(len(self.memory) * percentage_to_clear)
            for _ in range(num_to_clear):
                if self.memory:
                    self.memory.popleft() # removes the oldest transition from the memory queue
                    # where as pop() removes the newest

    def append(self, transition):
        # // append a transition data tuple to the memory queue
        # // the tuple is: (state, action, new_state, reward, terminated)
        self.memory.append(transition)

    def sample(self, sample_size):
        # // zip(tuple) is used to transpose the list of tuples into a tuple of lists
        state, action, next_state, reward, terminated = zip(*random.sample(self.memory, sample_size))
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        reward = np.array(reward)
        terminated = np.array(terminated)
        state = torch.FloatTensor(state).to(device='cuda' if torch.cuda.is_available() else 'cpu')
        action = torch.LongTensor(action).to(device='cuda' if torch.cuda.is_available() else 'cpu')
        next_state = torch.FloatTensor(next_state).to(device='cuda' if torch.cuda.is_available() else 'cpu')
        reward = torch.FloatTensor(reward).to(device='cuda' if torch.cuda.is_available() else 'cpu')
        terminated = torch.FloatTensor(terminated).to(device='cuda' if torch.cuda.is_available() else 'cpu')

        # // convert the lists of tensors to a batch-aligned dimension tensor for each batch
        return (state, action, next_state, reward, terminated)
    
    def __len__(self):
        return len(self.memory)
    
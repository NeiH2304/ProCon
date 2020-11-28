import numpy as np
import random
from collections import deque


class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.maxSize = max_size
        self.len = 0
        self.mem_size = max_size

    def store_transition(self, state, action, reward, next_state):
        """
		adds a particular transaction in the memory buffer
        """
        transition = (state, action, reward, next_state)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        batch_size = min(batch_size, self.len)
        batch = random.sample(self.buffer, batch_size)
        
        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        ns_arr = np.float32([arr[3] for arr in batch])
        
        return s_arr, a_arr, r_arr, ns_arr
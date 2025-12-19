import random
from collections import deque, namedtuple
import torch

Transition = namedtuple(
    "Transition",
    ["coords", "mask", "current_city", "action", "reward", "next_mask", "next_current_city", "done"],
)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.memory = dequeu(maxlen=capacity)
        self.device = device
    
    def __len__(self):
        return len(self.memory)
    
    def push(self, coords, mask, current_city, action, reward, next_mask, next_current_city, done):
        transition = Transition(
            coords.cpu(), mask.cpu(), current_city.cpu(), int(action), float(reward), next_mask.cpu(), next_current_city.cpu(), bool(done)
        )
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        coords = torch.stack(batch.coords).to(self.device)
        mask = torch.stack(batch.mask).to(self.device)
        current_city = torch.stack(batch.current_city).to(self.device)
        action = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        reward = torch.tensor(batch.reward, dtype=torch.float, device=self.device)
        next_mask = torch.stack(batch.next_mask).to(self.device)
        next_current_city = torch.stack(batch.next_current_city).to(self.device)
        done = torch.tensor(batch.done, dtype=torch.bool, device=self.device)

        return coords, mask, current_city, action, reward, next_mask, next_current_city, done

class EpsilonScheduler:
    def __init__(self, eps_start=1.0, eps_end=0.05, decay_steps=50000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = decay_steps
        self.steps_done = 0

    def step(self):
        self.steps_done += 1

    @property
    def epsilon(self):
        frac = min(self.steps_done / float(self.decay_steps), 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * frac
    

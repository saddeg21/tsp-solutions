import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
import time

class A3CNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim = 128):
        super(A3CNetwork, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_dim), # state_size -> 128
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 128 -> 128
            nn.ReLU(),
        )

        # actor layer represents the policy, generates logits for each action
        self.actor = nn.Linear(hidden_dim, action_size)  # 128 -> action_size
        # action times output

        # critic layer predicts states value V(s), whenever im in that state, how much reward can I expect
        self.critic = nn.Linear(hidden_dim, 1)  # 128 -> 1
        # 1 output value, ecause gives prediction of expected reward
    
    def forward(self, state):
        features = self.shared(state)

        policy_logits = self.actor(features)

        value = self.critic(features)

        return policy_logits, value

    def get_action(self, state, deterministic=False):
        """
        Flow

        1. State -> Network -> policy_logits,value
        2. Logits -> Softmax -> prob dist
        3. Get action from prob dist

        Deterministic:
        If deterministic, choose action with highest probability
        Else, sample from the distribution
        """

        with torch.no_grad():
            policy_logits, value = self.forward(state)

            probs = torch.softmax(policy_logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
            
        return action.item(), value.item()
    

class WorkerAgent:
    """
    ITS ABOUT OS

    A3C uses multiple workers to explore the environment in parallel.

    Worker1 : On episode 1, learning
    Worker2 : On episode 2, learning
    Each worker has its own copy of the network and interacts with its own instance of the environment.

    BUT ALL UPDATES THE GLOBAL NETWORK
    """

    def __init__(self, worker_id, global_network, optimizer, env_fn, gamma, max_steps, entropy_coef, value_coef):
        """
        worker_id: Unique ID for the worker
        global_network: a3C :D
        optimizer: GLOBAL optimizer, all workers have same optimizer to learn together
        env_fn: function to create a new environment instance (for parallel work)
        gamma: discount factor -> 0-1 and it calculates present value of future rewards
        max_steps: max steps per episode -> to prevent infinite episodes , the bigger it gets the longer it takes to train
        entropy_coef: coefficient for entropy regularization ( helps to exploration ) -> exploration vs exploitation
        value_coef: value loss weight -> helps to determine critic networks learning speed, balancing actor and critic updates
        """
        self.worker_id = worker_id
        self.global_network = global_network
        self.optimizer = optimizer
        self.env = env_fn()
        self.gamma = gamma
        self.max_steps = max_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.local_network = A3CNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n
        )

        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def compute_returns(self, rewards, next_value):
        returns = []

        R = next_value # the last states reward value -> bootstrap value

        # From reverse, traverse rewards
        for reward in reversed(rewards):
            # Calculate return for every step
            # Because every steps return depends on next steps return
            # G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + γ³*r_{t+3} + ...
            #     = r_t + γ*(r_{t+1} + γ*r_{t+2} + γ*r_{t+3} + ...)
            #     = r_t + γ*G_{t+1}
            R = reward + self.gamma * R
            # ınsert at beginning, because we are going reverse
            returns.insert(0, R)
        return returns
    
    def train_step(self):
        """
        Flow is like

        1. Sync local network with global network
        2. play as much as max_steps, and get experience
        3. compute returns
        4. compute advantages (return - value)
        5. compute losses (actor + critic - entropy_bonus)
        6. backpropagate losses to local network
        7. push local gradients to global network
        """

        self.local_network.load_state_dict(self.global_network.state_dict())

        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        entropies = []

        # reset environment, start episode
        state, _ = self.env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        # loop until done or max_steps reached
        for _ in range(self.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            policy_logits, value = self.local_network(state_tensor)

            probs = torch.softmax(policy_logits, dim=-1)
            dist = Categorical(probs)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            states.append(state)
            actions.append(action.item())
            values.append(value.squeeze().item())
            log_probs.append(log_prob)
            entropies.append(entropy)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            rewards.append(reward)
            episode_reward += reward
            step_count += 1

            if done:
                break

            state = next_state
        
        # calculate bootstrap value, if episode over its 0, else get value pred from critic
        if not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, next_value = self.local_network(state_tensor)
            next_value = next_value.squeeze().item()
        else:
            next_value = 0.0
        
        # calculate returns and advantages
        returns = self.compute_returns(rewards, next_value)
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(values)
        advantages = returns - values

        # compute losses
        policy_loss = []
        value_loss = []

        for log_prob, advantage, entropy, return_val, value in zip(log_probs, advantages, entropies, returns, values):
            policy_loss.append(-log_prob * advantage.detach())
            value_loss.append(nn.functional.mse_loss(value.unsqueeze(0), return_val.unsqueeze(0)))
        
        policy_loss = torch.stack(policy_loss).sum()
        value_loss = torch.stack(value_loss).sum()
        entropy_loss = -torch.stack(entropies).sum() # negative for maximizing

        total_loss = (policy_loss +
                      self.value_coef * value_loss +
                      self.entropy_coef * entropy_loss)
        
        # backpropagate local network
        self.optimizer.zero_grad()
        total_loss.backward()

        #gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), 40)

        # push local gradients to global network
        for local_param, global_param in zip(self.local_network.parameters(), self.global_network.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad
            else:
                global_param.grad = local_param.grad.clone()
        
        # update global optimizer
        self.optimizer.step()

        return total_loss.item(), episode_reward, step_count


def main():
    network = A3CNetwork(state_size=4, action_size=2)
    state = torch.randn(1, 4)
    policy_logits, value = network(state)

    print(policy_logits.shape)
    print(value.shape),

    action, value = network.get_action(state, deterministic=False)
    print(f"Action: {action}, Value: {value}")  

if __name__ == "__main__":
    main()


"""
NOTES

Bootstrap Value:
RL'de bir durumun ait değeri geleceği beklemeden, başka bir tahminle güncellemektir. Bu onu monte carlo'dan ayırır, çünkü monte carlo tamamen gerçek ödüllere dayanır. Bootstraping, gelecekteki ödülleri tahmin etmek için mevcut değer tahminlerini kullanır.
Geçen hafta 100tl idi o zaman 20 tl artmıştır, şuan 120tldir

Episode bitmeden öğrenmeye yarar, tek bir transition bile update yapmaya yeter

Online öğrenme için çok uygundur.

Temporal Difference buna dayanır

------------------------------------------------------------------------------------
compute_returns fonksiyonu, 
gelecekteki ödülleri dikkate alarak her adım için toplam beklenen getiriyi hesaplar.
R = reward + self.gamma * R yani gerçek reward + discountlanmış gelecekteki reward

# Worker bir episode oynadı:
rewards = [1.0, 1.0, 1.0]  # ← GERÇEK: Environment verdi
next_value = 5.0           # ← TAHMİN: Critic'ten geldi

# compute_returns:
R = 5.0  # Bootstrap (TAHMİN)

# Iteration 1: reward=3 (GERÇEK)
R = 1.0 + 0.99 * 5.0 = 5.95  # ← GERÇEK + TAHMİN
returns = [5.95]

# Iteration 2: reward=2 (GERÇEK)
R = 1.0 + 0.99 * 5.95 = 5.89  # ← GERÇEK + TAHMİN
returns = [5.89, 5.95]

# Iteration 3: reward=1 (GERÇEK)
R = 1.0 + 0.99 * 5.89 = 5.83  # ← GERÇEK + TAHMİN
returns = [5.83, 5.89, 5.95]

Episode bitince ise 
# Episode sonu: Artık gelecek yok
next_value = 0.0  # ← TAHMİN DEĞİL, GERÇEK (hiç gelecek yok)

rewards = [1.0, 1.0, 0.0]  # last step
returns = compute_returns(rewards, 0.0)
# = [1.99, 0.99, 0.0]
# ↑ Tamamen GERÇEK rewards'a dayalı (Monte Carlo gibi)

"""
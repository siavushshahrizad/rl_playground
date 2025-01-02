import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

class Env():
    def __init__(self, num_pages=10, num_advertisements=10, steps_remaining=5000) -> None:
        self.state = random.randint(0, num_pages - 1)
        self.num_actions = num_advertisements
        self.num_states = num_pages
        self.payout_probs = np.random.rand(num_pages, num_advertisements)
        self.steps_remaining = steps_remaining - 1

    def get_state(self) -> int:
        return self.state
    
    def transition(self, num_pages) -> None:
        self.state = random.randint(0, num_pages - 1)
    
    def get_payout_probs(self, page: int, ad: int) -> float:             
        return self.payout_probs[page][ad]                      
    
    def get_num_actions(self) -> int:                              
        return self.num_actions

    def get_num_states(self) -> int:
        return self.num_states
    
    def is_done(self) -> bool:
        return self.steps_remaining == 0
    
    def decrement_steps(self) -> None:
        self.steps_remaining -= 1

    def reward(self, state: int, action: int,) -> int: 
        num = random.random()
        prob = self.get_payout_probs(state, action)
        result = int(num < prob)
        return result
    
class Agent(nn.Module):
    def __init__(self, num_states, num_actions, hidden_in=128, hidden_out=64):
        super().__init__()
        self.fc1 = nn.Linear(num_states, hidden_in)
        self.h1 = nn.Linear(hidden_in, hidden_out)
        self.fc2 = nn.Linear(hidden_out, num_actions)

    def make_state_readable(self, curr_state: int, num_states) -> torch.tensor:
        one_hot = torch.zeros(num_states)
        one_hot[curr_state] = 1
        one_hot = one_hot.unsqueeze(0)                  # Best practice as forward function expects a dimension for batches
        return one_hot

    def forward(self, state: torch.tensor) -> torch.tensor:
        preds = F.relu(self.fc1(state))
        preds = F.relu(self.h1(preds))
        preds = self.fc2(preds)
        return preds
    
    def soft_max(self, preds: torch.tensor, tau=0.1) -> torch.tensor:
        exp = torch.exp(preds / tau)
        probs = exp / torch.sum(exp)
        return probs
    
    def select_action(self, preds: torch.tensor) -> int:
        probs = self.soft_max(preds)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def train(self, predicted_reward: float, actual_reward: int, optimizer):
        optimizer.zero_grad()
        squared_loss = (predicted_reward - actual_reward) ** 2
        squared_loss.backward() 
        optimizer.step()
    
if __name__ == "__main__":
    env = Env()
    num_states = env.get_num_states()
    num_actions = env.get_num_actions()
    agent = Agent(num_states, num_actions)
    optimizer = optim.Adam(agent.parameters(), lr=0.01)

    reward = 0.0
    step = 1
    avg_rewards = []

    while not env.is_done():
        curr_state = env.get_state()
        readable_state = agent.make_state_readable(curr_state, num_states)
        preds = agent.forward(readable_state)
        action = agent.select_action(preds)
        curr_reward = env.reward(curr_state, action)
        reward += curr_reward
        avg_rewards.append(reward / step)
        agent.train(preds[0][action], curr_reward, optimizer)
        env.decrement_steps()        
        step += 1
        env.transition(num_states)

    print("Reward achieved is : ", reward)
    plt.plot(avg_rewards)
    plt.title("Average clicks for ads")
    plt.xlabel("Num ads shown")
    plt.ylabel("Average clicks")
    plt.show()
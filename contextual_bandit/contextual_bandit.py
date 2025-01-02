import numpy as np
import torch
import random
import matplotlib.pyplot as plt

ARMS = 10
BATCH_SIZE = 1
DIMENIONS_IN = ARMS
HIDDEN_SIZE = 100
DIMENSIONS_OUT = ARMS

class ContextBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.init_distribution(arms)
        self.update_state()

    def init_distribution(self, arms):
        self.bandit_matrix = np.random.rand(arms, arms)

    def update_state(self):
        self.state = np.random.randint(0, self.arms)

    def get_state(self):
        return self.state

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward
    
model = torch.nn.Sequential(
    torch.nn.Linear(DIMENIONS_IN, HIDDEN_SIZE),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_SIZE, DIMENSIONS_OUT),
    torch.nn.ReLU(),
)

loss_fn = torch.nn.MSELoss()

def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec

def running_mean(x, N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y

def softmax(av, tau=1.12):
    softm = (np.exp(av / tau) / np.sum(np.exp(av / tau)))
    return softm

def train(env, epochs=5000, learning_rate=1e-2):
    cur_state = torch.Tensor(one_hot(ARMS, env.get_state()))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []
    for i in range(epochs):
        y_pred = model(cur_state)
        if env.get_state() == 0 and i % 100 == 0:
            print("Epoch: ", i)
            print("State:", env.get_state())
            print(y_pred)
            print(  )
        av_softmax = softmax(y_pred.data.numpy(), tau=2.0) 
        av_softmax /= av_softmax.sum() 
        choice = np.random.choice(ARMS, p=av_softmax) 
        cur_reward = env.choose_arm(choice) 
        one_hot_reward = y_pred.data.numpy().copy() 
        one_hot_reward[choice] = cur_reward 
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward)
        loss = loss_fn(y_pred, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(one_hot(ARMS, env.get_state()))
    return np.array(rewards)

if __name__ == "__main__":
    env = ContextBandit(ARMS)
    state = env.get_state()
    rewards = train(env)
    plt.plot(running_mean(rewards,N=500))
    plt.xlabel('Epochs')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.show()

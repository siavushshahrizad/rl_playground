import random
import matplotlib.pyplot as plt

EPSILON = 0.2

class Environment:
    def __init__(self, n):
        self.steps_remaining = 10000
        self.num_slot_machines = n
        self.payout_probabilities = [random.random() for _ in range(self.num_slot_machines)]
        self.observed_payout_average = [0 for _ in range(self.num_slot_machines)]
        self.num_played = [0 for _ in range(self.num_slot_machines)]

    def get_action(self):
      if random.random() < EPSILON:
        return random.randint(0, self.num_slot_machines - 1)
      
      max_payout = float('-inf')
      best_action = None

      for idx, num in enumerate(self.observed_payout_average):
        if num > max_payout:
          max_payout = num
          best_action = idx

      return best_action

    def action(self, action: int):
      reward = 1 if random.random() <= self.payout_probabilities[action] else 0       # Pays out $1 if successful

      # Update record
      new_payout_average = (self.observed_payout_average[action] * self.num_played[action] + reward) \
                            / (self.num_played[action] + 1)
      self.observed_payout_average[action] = new_payout_average
      self.num_played[action] += 1

      self.steps_remaining -= 1
      return reward
    
    def is_done(self):
        return self.steps_remaining == 0
    

class Agent:
    def __init__(self):
        self.reward = 0.0
        self.steps = 0
        self.avg_reward = []

    def step(self, env: Environment):
        self.steps += 1
        action = env.get_action()
        reward = env.action(action)
        self.reward += reward
        self.avg_reward.append(self.reward / self.steps)

if __name__ == "__main__":
    env = Environment(10)
    agent = Agent()
   
    
    while not env.is_done():
      agent.step(env)

    print("The cumulative reward is: ", agent.reward)
    print("The historical payouts were: ", env.observed_payout_average)
    print("Having paid out, this many times: ", env.num_played)

    plt.plot(agent.avg_reward)
    plt.title("Avg rewards over time")
    plt.xlabel("Plays")
    plt.ylabel("Avg reward")
    plt.show()
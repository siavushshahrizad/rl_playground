"""
A simplified environment-agent file. The agent takes a random action 
and gets random rewards. The file only exists to exemplify some of the
basic agent-environment code. 

At a minimum RL needs an agent and an environment. This environment
must allow the agent to take actions and it must yield rewards. 
Later the agent needs a good plan how to maximise rewards. 
"""
import random 

class Environment:
    def __init__(self):
        self.steps = 10
    
    def get_observation(self):              # This is to figure out the state; here it does nothing. 
        return [0.0, 0.0, 0.0]
    
    def get_actions(self):
        return [0, 1]

    def is_done(self):
        return self.steps == 0  
    
    def action(self, action: int):                           # Takes an action and returns reward
        if self.is_done():
            raise Exception("Game is over")
        self.steps -= 1
        return random.random()
    
class Agent:
    def __init__(self):
        self.reward = 0.0

    def step(self, env: Environment):
        current_obs = env.get_observation()                 # You can see in this demo, this isn't use at all. 
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.reward += reward

if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward: ", agent.reward)
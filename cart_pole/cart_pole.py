import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.HumanRendering(env)
    total_reward = 0.0 
    total_steps = 0

    obs, _ = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, is_done, is_truncated, _ = env.step(action)
        total_reward += reward
        total_steps += 1

        if is_done or is_truncated:
            break

    print("The cumulative reward was: ", total_reward, "and the agent took ", total_steps, "steps.")
    env.close()
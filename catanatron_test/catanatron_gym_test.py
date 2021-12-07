import random
import gym

env = gym.make("catanatron_gym:catanatron-v0")
observation = env.reset()
for _ in range(1000):
    action = random.choice(env.get_valid_actions()) # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    print(observation)
    print()
    print(reward)
    print()
    print(done)
    print()
    print(info)
    print()
    exit()
    if done:
        observation = env.reset()
env.close()

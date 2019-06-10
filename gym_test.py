


import gym
import minerl

import logging
import matplotlib.pyplot as plt


def main():
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make("CartPole-v1")
    output = env.reset()
    if isinstance(output, tuple):
        obs, _ = output
    else:
        obs = output
        if obs.shape[-1] != 3:
            obs = env.render('rgb_array')
    done = False
    net_reward = 0

    plt.interactive(True)
    plt.imshow(obs)
    plt.show()

    while not done:
        action = env.action_space.sample()

        obs, reward, done, info = env.step(
            action)

        if not isinstance(obs, tuple):
            if obs.shape[-1] != 3:
                obs = env.render('rgb_array')

        net_reward += reward
        print("Total reward: ", net_reward)
        plt.imshow(obs)
        plt.show()
        plt.pause(0.001)

    plt.interactive(False)







if __name__ == "__main__":
    main()

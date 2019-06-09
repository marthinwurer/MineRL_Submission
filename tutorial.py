

import gym
import minerl

import logging
import matplotlib.pyplot as plt


def main():
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make('MineRLObtainDiamondDense-v0')
    obs, _ = env.reset()
    done = False
    net_reward = 0

    plt.interactive(True)
    plt.imshow(obs["pov"])
    plt.show()

    while not done:
        action = env.action_space.sample()

        obs, reward, done, info = env.step(
            action)

        net_reward += reward
        print("Total reward: ", net_reward)
        plt.imshow(obs["pov"])
        plt.show()
        plt.pause(0.001)

    plt.interactive(False)







if __name__ == "__main__":
    main()

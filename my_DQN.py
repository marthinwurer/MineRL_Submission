import logging

import torch
import torch.nn.functional as F
import gym
from torch import nn, Tensor

from DQN import DQNNet, DQNHyperparameters, DQNTrainingState
from utilities import GenericConvolutionalEncoder, flat_shape, flatten, GenericFullyConnected

logger = logging.getLogger(__name__)


class MaitlandDQN(DQNNet):

    def __init__(self, input_space, action_space, fc_total=128, activation=F.relu):
        super().__init__(input_space, action_space)
        logger.debug("Input shape type: %s" % type(self.input_shape))
        logger.debug("New MaitlandDQN with Input Shape: %s" % (self.input_shape,))

        self.conv_layers = GenericConvolutionalEncoder(self.input_shape)
        self.activation = activation

        final_shape = flat_shape(self.conv_layers.output_shape)

        self.fc = nn.Linear(final_shape, fc_total)
        self.to_actions = nn.Linear(fc_total, self.action_shape)

    def forward(self, x: Tensor):
        """
        Expects (N,Cin,H,W)
        N is batch size
        Args:
            x:

        Returns:

        """
        # make sure that the input shape is correct
        in_shape = x.shape[1:]
        if in_shape != self.input_shape:
            raise AssertionError("%s != %s" % (in_shape, self.input_shape))
        # convert the input image to the correct format
        x = x.to(torch.float32) / 255

        x = self.conv_layers(x)
        x = self.activation(self.fc(flatten(x)))
        x = self.to_actions(x)
        return x


class FCDQN(DQNNet):

    def __init__(self, input_shape, num_actions, fc_total=128, activation=F.relu):
        super().__init__(input_shape, num_actions)

        in_len = self.input_shape[0]

        self.layers = GenericFullyConnected(in_len, fc_total, 3, activation=activation)
        self.activation = activation

        self.to_actions = nn.Linear(fc_total, self.action_shape)

    def forward(self, x: Tensor):
        # make sure that the input shape is correct
        assert x.shape[1:] == self.input_shape
        x = x.to(torch.float32)
        x = self.layers(x)
        x = self.to_actions(x)
        return x


if __name__ == "__main__":
    FORMAT = '%(asctime)-15s | %(filename)s:%(lineno)s | %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper = DQNHyperparameters(batch_size=128)
    env = gym.make("CarRacing-v0")
    state = DQNTrainingState(MaitlandDQN, env, device, hyper)

    print(env.get_observation_shape())
    exit()

    state.train_for_episodes(500)

    state.save_model("saved_nets/basic_doom.mod")







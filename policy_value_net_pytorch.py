# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """

    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        # print("debug:", self.policy_value_net)
        """
        debug: Net(
          (conv1): Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act_conv1): Conv2d(128, 4, kernel_size=(1, 1), stride=(1, 1))
          (act_fc1): Linear(in_features=144, out_features=36, bias=True)    # 4 x width x height
          (val_conv1): Conv2d(128, 2, kernel_size=(1, 1), stride=(1, 1))
          (val_fc1): Linear(in_features=72, out_features=64, bias=True)    # 2 x width x height
          (val_fc2): Linear(in_features=64, out_features=1, bias=True)
        )
        """
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            # debug: <class 'collections.OrderedDict'>
            # print("debug:", type(net_params))
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            # print("debug: pv", type(state_batch), state_batch.shape)
            # print("debug: pv", type(log_act_probs), log_act_probs.shape, log_act_probs[0])
            # print("debug: pv", type(value), value.shape, value[0])
            # print("debug: pv", type(act_probs), act_probs.shape, act_probs[0])
            """
            debug: pv <class 'torch.Tensor'> torch.Size([512, 4, 6, 6])
            debug: pv <class 'torch.Tensor'> torch.Size([512, 36]) tensor([-3.5188, -3.6358, -3.5779, -3.6464, -3.6030, -3.6298, -3.5478,
                    -3.5090, -3.5997, -3.5677, -3.5541, -3.6722, -3.5616, -3.5636,
                    -3.5926, -3.4936, -3.5709, -3.6210, -3.5447, -3.6076, -3.5882,
                    -3.5600, -3.4815, -3.5765, -3.6788, -3.6113, -3.5063, -3.6241,
                    -3.5781, -3.5612, -3.5779, -3.6497, -3.6608, -3.6400, -3.5247,
                    -3.6140])
            debug: pv <class 'torch.Tensor'> torch.Size([512, 1]) tensor(1.00000e-02 *
                   [ 2.5594])
            debug: pv <class 'numpy.ndarray'> (512, 36) [0.02963571 0.02636158 0.02793355 0.02608529 0.02724108 0.02652155
             0.02878688 0.0299259  0.02733225 0.02821934 0.02860781 0.02542085
             0.02839345 0.02833655 0.02752793 0.03039031 0.02813084 0.02675619
             0.0288761  0.02711727 0.02764767 0.02843794 0.03076018 0.02797232
             0.02525269 0.02701536 0.03000749 0.02667333 0.02792769 0.02840393
             0.0279335  0.02599909 0.02571266 0.02625342 0.02945924 0.02694306]
            """
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())

        # print("debug: pvf", legal_positions)
        # print("debug: pvf,", value, log_act_probs, act_probs)
        """
        debug: pvf [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35]
        debug: pvf, tensor([[-0.1340]]) tensor([[-3.5891, -3.6159, -3.6483, -3.6113, -3.5677, -3.6311, -3.5342,
                 -3.6449, -3.6345, -3.5444, -3.5193, -3.5488, -3.5690, -3.6075,
                 -3.6792, -3.6329, -3.5296, -3.6544, -3.5786, -3.5870, -3.5027,
                 -3.5944, -3.5537, -3.6383, -3.6251, -3.5364, -3.5302, -3.6542,
                 -3.6004, -3.6269, -3.5623, -3.5152, -3.5181, -3.5594, -3.5809,
                 -3.5221]]) [0.02762223 0.02689152 0.02603547 0.02701722 0.02822023 0.02648623
         0.02918113 0.02612401 0.02639725 0.02888605 0.02961869 0.02876004
         0.02818486 0.02711975 0.02524304 0.0264391  0.02931764 0.02587646
         0.02791596 0.02768259 0.030115   0.02747608 0.0286187  0.0262969
         0.02664538 0.02911758 0.02930048 0.02588317 0.02731332 0.02659828
         0.02837263 0.02974232 0.02965699 0.02845607 0.02785116 0.02953648]
        """
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        # # debug: pvf value tensor(-0.1340)
        # print("debug: pvf value", value)
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        # print("debug:", loss, loss.data[0], "{:.5f} |x {:.5f}".format(loss.data[0], loss))
        # print("debug:", entropy, entropy.data[0])
        """
            debug: tensor(4.1732) tensor(4.1732) 4.17323 |x 4.17323
            debug: tensor(3.5791) tensor(3.5791)
        """
        return loss.data[0], entropy.data[0]

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)

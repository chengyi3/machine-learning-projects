import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils
from policies import QPolicy

#inspired by https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html
#and https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html
#and https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/13
# and https://towardsdatascience.com/deep-reinforcement-learning-tutorial-with-open-ai-gym-c0de4471f368
def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    model = nn.Sequential(
          # nn.Conv2d()
          #64,32
          nn.Linear(statesize, 18),
          nn.ReLU(),
          nn.Linear(18,10),
          nn.ReLU(),
          nn.Linear(10,actionsize)
        )
    return model


class DQNPolicy(QPolicy):
    """
    Function approximation via a deep network
    """

    def __init__(self, model, statesize, actionsize, lr, gamma):
        """
        Inititalize the dqn policy

        @param model: the nn.Module instance returned by make_dqn
        @param statesize: dimension of the input continuous state space.
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate
        @param gamma: discount factor
        """
        super().__init__(statesize, actionsize, lr, gamma)
        self.model = model
        self.statesize = statesize
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma

    def qvals(self, state):
        """
        Returns the q values for the states.

        @param state: the state

        @return qvals: the q values for the state for each action.
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.from_numpy(state).type(torch.FloatTensor)
            qvals = self.model(states)
        return qvals.numpy()

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        optimizer = torch.optim.SGD(self.model.parameters(), self.lr)
        optimizer.zero_grad()
        x = (torch.from_numpy(state)).float()
        s = self.model(x)
        p = s[action]
        target = 0
        if done == True:
            target = reward
        else:
            y = (torch.from_numpy(next_state)).float()
            # print(self.model(y))
            target = (reward + self.gamma * torch.max(self.model(y))).item()
            # print(target)

            # print(torch.max(self.model(y)))
        # f = nn.MSELoss()
        # print(p.size())
        # print(target.size())
        # loss = f(p, target.item())

        loss = ((p - target)**2)
        # l = loss
        # print(loss)
        loss.backward()
        # print(loss)
        optimizer.step()
        # weights = self.model.weight
        # loss = torch.nn.MSELoss()

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.3)
        # optimizer.zero_grad()
        # loss = 0
        # l = 0
        # if done == True:
        #      target = reward
        #      for i in len(weights):
        #          weights[i] = weights[i]+self.lr * (target - weights[i])
        #      l = loss(self.model,target)
        #      # loss.backward()
        # else:
        #     target = reward + self.gamma * np.max(self.qvals(next_state))
        #     for i in len(weights):
        #         weights[i] = weights[i]+self.lr * (target - weights[i])
        #     l = loss(self.model,target)
            # loss = ((self.model[state] - target)**2)
            # loss.backward()


         # loss_fn(model(input), target).backward()
        # optimizer.step()
        # d = self.discretize(state)
        # Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action]

        return loss.item()

    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    policy = DQNPolicy(make_dqn(statesize, actionsize), statesize, actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'models/dqn.model')

import numpy as np
from collections import namedtuple
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical, Normal
from torch.nn.functional import smooth_l1_loss

"""
Contains the definition of the agent that will run in an
environment.
"""


######################################################################
############################### RANDOM ###############################


class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        return np.random.normal(0, 10)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass


#######################################################################
########################### POLICY GRADIENT ###########################


class PolicyGradientModule(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(PolicyGradientModule, self).__init__()
        self.hidden = torch.nn.Linear(n_input, 128)  # hidden layer
        self.predict = torch.nn.Linear(128, n_output)  # output layer

    def forward(self, state):
        x = F.relu(self.hidden(state))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


class PolicyGradientAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 1
        self.decay_rate = 4
        self.ngames = 0
        self.max_decay = 200 ** self.decay_rate

        self.policy_history = []
        self.reward_history = []

        self.learning_rate = 0.1
        self.gamma = 0.99

        self.state_dim = 2
        self.n_output = 1
        self.model = PolicyGradientModule(self.state_dim, self.n_output)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.eps = np.finfo(np.float32).eps.item()

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        # TODO : Figure out how the above is different from environment.reset()

        if len(self.reward_history) > 0:
            self.ngames += 1
            self.epsilon = 1 - (self.ngames ** self.decay_rate) / self.max_decay
            print('epsilon :', self.epsilon)

            # Update network weights
            R = 0
            policy_loss = []
            rewards = []
            for r in self.reward_history[::-1]:
                R = r + self.gamma * R
                rewards.insert(0, R)
            rewards = torch.tensor(rewards)

            actions = torch.tensor(self.policy_history)
            for action, reward in zip(actions, rewards):
                policy_loss.append(-action * reward)

            self.optimizer.zero_grad()
            policy_loss = [Variable(loss, requires_grad=True) for loss in policy_loss]
            policy_loss = torch.stack(policy_loss).sum()
            print('Total loss :', policy_loss)
            policy_loss.backward()
            self.optimizer.step()

            self.policy_history = []
            self.reward_history = []

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if np.random.rand() <= self.epsilon:
            action = np.random.normal(0, 10)
        else:
            action = self.learned_act(observation)
        return action

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.policy_history.append(action)
        self.reward_history.append(reward)

    def learned_act(self, observation):
        state = torch.from_numpy(np.asarray(observation)).type(torch.FloatTensor)
        action = self.model(state).item()
        return action


######################################################################
######################### BASIC ACTOR CRITIC #########################


class ActorCriticModule(nn.Module):
    def __init__(self, state_dim, n_output):
        super(ActorCriticModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        # self.fc2 = nn.Linear(64, 128)

        self.action_head = nn.Linear(128, n_output)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))

        probs = F.softmax(self.action_head(x))
        value = self.value_head(x)
        return probs, value


class BasicActorCriticAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 1
        self.decay_rate = 4
        self.ngames = 0
        self.max_decay = 200 ** self.decay_rate

        self.policy_history = []
        self.reward_history = []
        self.saveAction = namedtuple('SavedActions', ['probs', 'action_values'])

        self.learning_rate = 0.001
        self.gamma = 0.99

        self.state_dim = 2
        self.num_actions = 100

        self.model = ActorCriticModule(self.state_dim, self.num_actions)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.eps = np.finfo(np.float32).eps.item()

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """

        if len(self.reward_history) > 0:
            self.ngames += 1
            self.epsilon = 1 - (self.ngames ** self.decay_rate) / self.max_decay
            print('epsilon :', self.epsilon)

            # Update network weights
            rewards = []
            policy_loss = []
            value_loss = []
            R = 0

            for r in self.reward_history[::-1]:
                R = r + self.gamma * R
                rewards.insert(0, R)

            rewards = torch.tensor(rewards)

            # Figure out loss
            for (log_prob, value), r in zip(self.policy_history, rewards):
                reward = r.item() - value
                policy_loss.append(-log_prob * reward)
                value_loss.append(smooth_l1_loss(torch.tensor(value), r))

            self.optimizer.zero_grad()
            policy_loss = [Variable(torch.tensor(loss), requires_grad=True) for loss in policy_loss]
            loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
            print('Total loss :', loss)
            loss.backward()
            self.optimizer.step()

            self.policy_history = []
            self.reward_history = []

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if np.random.rand() <= self.epsilon:
            action = np.random.normal(0, 10)
            self.policy_history.append(self.saveAction(1, action))
        else:
            action = self.learned_act(observation)
        return action

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.reward_history.append(reward)

    def learned_act(self, observation):
        state = torch.from_numpy(np.asarray(observation)).type(torch.FloatTensor)
        probs, value = self.model(state)
        c = Categorical(probs)
        action = c.sample()
        log_prob = c.log_prob(action)
        self.policy_history.append(self.saveAction(log_prob, value[0]))
        action = 30 * (action.item() - self.num_actions / 2) / self.num_actions
        return action


######################################################################
############################ ACTOR CRITIC ############################


class ActorModule(nn.Module):
    def __init__(self, n_input, n_output):
        super(ActorModule, self).__init__()
        hidden_dim = 512
        self.log_std_min = -20
        self.log_std_max = 2

        self.linear1 = nn.Linear(n_input, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear3 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.linear4 = nn.Linear(int(hidden_dim/4), int(hidden_dim/8))
        self.linear5 = nn.Linear(int(hidden_dim / 8), int(hidden_dim / 16))

        self.log_std_min = self.log_std_min
        self.log_std_max = self.log_std_max
        self.mean_linear = nn.Linear(int(hidden_dim/16), n_output)
        self.log_std_linear = nn.Linear(int(hidden_dim/16), n_output)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std


class CriticModule(nn.Module):
    def __init__(self, n_input, n_output):
        super(CriticModule, self).__init__()
        hidden_dim = 512
        self.linear1 = nn.Linear(n_input, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear3 = nn.Linear(int(hidden_dim/2), int(hidden_dim / 4))
        self.linear4 = nn.Linear(int(hidden_dim / 4), int(hidden_dim / 8))
        self.linear5 = nn.Linear(int(hidden_dim/8), n_output)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, n_input, n_output):
        super(SoftQNetwork, self).__init__()
        hidden_dim = 512
        self.linear1 = nn.Linear(n_input + n_output, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear3 = nn.Linear(int(hidden_dim/2), int(hidden_dim / 4))
        self.linear4 = nn.Linear(int(hidden_dim/4), 1)

    def forward(self, state, action):
        if len(action.size()) == 1:
            state_x = []
            state_vx = []
            for s in state:
                state_x.append(s[0].item())
                state_vx.append(s[1].item())
            state_x = torch.FloatTensor(state_x)
            state_vx = torch.FloatTensor(state_vx)
            x = torch.stack([state_x, state_vx, action], -1)
        else:
            x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class ActorCriticAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 1
        self.decay_rate = 0.99
        self.ngames = 0
        self.max_decay = 200 ** self.decay_rate

        self.game_history = Memory(400)
        self.last_index = -1
        self.won_games = Memory(20000)
        self.treshold = 50
        self.lost_games = Memory(100)
        self.batch_size = 4096

        self.learning_rate_actor = 0.00001
        self.learning_rate_critic = 0.0005
        self.learning_rate_q = 0.0001
        self.gamma = 1
        self.soft_tau = 1e-2

        self.state_dim = 2
        self.n_output = 1

        self.model_actor = ActorModule(self.state_dim, self.n_output)
        self.model_critic = CriticModule(self.state_dim, self.n_output)
        self.target_value_net = CriticModule(self.state_dim, self.n_output)
        self.soft_q_net = SoftQNetwork(self.state_dim, self.n_output)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.policy_optimizer = optim.Adam(self.model_actor.parameters(), lr=self.learning_rate_actor)
        self.value_optimizer = optim.Adam(self.model_critic.parameters(), lr=self.learning_rate_critic)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=self.learning_rate_q)

        #TODO : Delete these lines
        self.reward_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.soft_q_loss_history = []

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """

        if self.last_index >= 0:
            self.ngames += 1
            #self.epsilon = 1 - (self.ngames ** self.decay_rate) / self.max_decay
            self.epsilon = self.epsilon*self.decay_rate
            print('epsilon :', self.epsilon)

            # Save the game in won_games or lost_games
            reward = self.game_history.get_reward()
            #TODO : Delete below line
            self.reward_history.append(sum(reward))
            won = False
            for r in reward:
                if r >= 80:
                    won = True
                    break

            '''if won:
                for m in self.game_history.memory:
                    self.won_games.remember(m)
            '''
            if won and len(self.game_history.memory) <= self.treshold:
                for m in self.game_history.memory:
                    self.won_games.remember(m)
            elif won and len(self.game_history.memory) > self.treshold:
                for i in range(self.treshold):
                    self.won_games.remember(self.game_history.memory[-self.treshold+i])
            else:
                for m in self.game_history.memory:
                    self.lost_games.remember(m)

            '''if sum(reward) >= 0:
                for m in self.game_history.memory:
                    self.won_games.remember(m)
                print('won games size:', len(self.won_games.memory))
            else:
                for m in self.game_history.memory:
                    self.lost_games.remember(m)'''

            # Update network weights
            if self.batch_size <= len(self.won_games.memory):
                observation, action, reward, next_observation = self.won_games.sample(self.batch_size)
            elif len(self.won_games.memory) == 0 and self.batch_size > len(self.lost_games.memory):
                observation, action, reward, next_observation = self.lost_games.sample(len(self.lost_games.memory))
            elif len(self.won_games.memory) == 0 and self.batch_size <= len(self.lost_games.memory):
                observation, action, reward, next_observation = self.lost_games.sample(self.batch_size)
            elif len(self.lost_games.memory) == 0 and self.batch_size > len(self.won_games.memory):
                observation, action, reward, next_observation = self.won_games.sample(len(self.won_games.memory))
            elif len(self.lost_games.memory) == 0 and self.batch_size <= len(self.won_games.memory):
                observation, action, reward, next_observation = self.won_games.sample(self.batch_size)
            elif self.batch_size > len(self.won_games.memory) + len(self.lost_games.memory):
                observation, action, reward, next_observation = self.won_games.sample(len(self.won_games.memory))
                observation2, action2, reward2, next_observation2 = self.lost_games.sample(len(self.lost_games.memory))
                observation = np.append(observation, observation2, 0)
                action = np.append(action, action2)
                reward = np.append(reward, reward2)
                next_observation = np.append(next_observation, next_observation2, 0)
            else:
                observation, action, reward, next_observation = self.won_games.sample(len(self.won_games.memory))
                observation2, action2, reward2, next_observation2 = self.lost_games.sample(
                    self.batch_size - len(self.won_games.memory))
                observation = np.append(observation, observation2, 0)
                action = np.append(action, action2)
                reward = np.append(reward, reward2)
                next_observation = np.append(next_observation, next_observation2, 0)

            observation = torch.FloatTensor(observation)
            next_observation = torch.FloatTensor(next_observation)
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor(reward).unsqueeze(1)

            predicted_q_value = self.soft_q_net(observation, action)
            predicted_value = self.model_critic(observation)
            new_action, log_prob, epsilon, mean, log_std = self.model_actor.evaluate(observation)

            # Training Q Function
            target_value = self.target_value_net(next_observation)
            target_q_value = reward + self.gamma * target_value
            q_value_loss = self.soft_q_criterion(predicted_q_value, target_q_value.detach())
            #TODO : Delete below line
            self.soft_q_loss_history.append(q_value_loss.item())
            self.soft_q_optimizer.zero_grad()
            q_value_loss.backward()
            self.soft_q_optimizer.step()

            # Training Value Function
            predicted_new_q_value = torch.min(self.soft_q_net(observation,
                                                              new_action), self.soft_q_net(observation, new_action))
            target_value_func = predicted_new_q_value - log_prob
            value_loss = self.value_criterion(predicted_value, target_value_func.detach())
            #TODO : Delete below line
            print(value_loss)
            self.value_loss_history.append(value_loss.item())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Training Policy Function
            policy_loss = (log_prob - predicted_new_q_value).mean()
            # TODO : Delete below line
            self.policy_loss_history.append(policy_loss.item())
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            for target_param, param in zip(self.target_value_net.parameters(), self.model_critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
                )

            self.game_history.reset()
            self.last_index = -1

            # TODO : Delete below lines
            if self.ngames == 199:
                import pandas as pd
                import matplotlib.pyplot as plt
                data = pd.DataFrame({'policy loss': self.policy_loss_history, 'value loss': self.value_loss_history,
                                     'q value loss': self.soft_q_loss_history, 'reward': self.reward_history})
                data.plot()
                plt.show()

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if np.random.rand() <= self.epsilon:
            action = np.random.normal(0, 10)
        else:
            action = self.learned_act(observation)
        return action

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        if self.last_index >= 0:
            self.game_history.add_next_observation(self.last_index, observation)
        self.last_index = self.game_history.remember([observation, action, reward, observation])

    def learned_act(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0)
        mean, log_std = self.model_actor.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        action = (action.cpu())[0].detach()
        return action.item()


class Memory(object):
    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.memory = list()

    def remember(self, m):
        if len(self.memory) >= self.max_memory:
            del self.memory[0]
        self.memory.append(m)
        return len(self.memory) - 1

    def get_reward(self):
        _, _, reward, _ = map(np.stack, zip(*self.memory))
        return reward

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation = map(np.stack, zip(*batch))
        return observation, action, reward, next_observation

    def add_next_observation(self, index, observation):
        (self.memory[index])[-1] = observation

    def reset(self):
        self.memory = list()


Agent = ActorCriticAgent
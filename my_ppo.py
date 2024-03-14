import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
import threading

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.optim.lr_scheduler as lr_scheduler
from functools import partial

DROPOUT_PROB = 0.0
NUM_TRAINING_EPOCHS = 100000
NUM_PARALLEL_SIMS = 10
NUM_ACTIONS_PER_SIM = 1000
ENVIRONMENT = "Humanoid-v3"
BATCH_SIZE = 500
NUM_OPT_STEPS_PER_SAMPLE = 10
GAMMA = 0.99
LAMBDA = 0.97
VALUE_LR = 1e-3
POLICY_LR = 3e-4
LR_GAMMA = 0.999
PPO_EPSILON = 0.2
TARGET_KL = 100.05
NORMAL_STARTING_LOG_STD = -1.5
LOG_STD_DECREMENT_PER_EPOCH = 0
USE_SQUASHING_FUNCTION = False

env = gym.make(ENVIRONMENT)

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
env.seed(seed)

reward_scalar = 1.0
reward_offset = 0.0

policy_scalar = env.action_space.high - env.action_space.low
policy_offset = env.action_space.low + policy_scalar / 2.0
policy_scalar = torch.from_numpy(policy_scalar)
policy_offset = torch.from_numpy(policy_offset)

observation_scalar = torch.ones(env.observation_space.shape[0])
observation_offset = torch.zeros(env.observation_space.shape[0])

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def discount_cumsum(x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.

        input: 
            vector x, 
            [x0, 
            x1, 
            x2]

        output:
            [x0 + discount * x1 + discount^2 * x2,  
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOEpisodeMemory(object):

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.log_probs_of_actions = []

    def push(self, observation, action, reward, done, log_prob_of_action):
        self.observations.append(observation)
        self.actions.append(action.view(-1, 1))
        self.rewards.append(reward)
        self.done.append(done)
        self.log_probs_of_actions.append(log_prob_of_action)
    
    def get_generalized_advantage_estimates(self, gamma, lamda, value_net):
        '''Returns a GAE estimate for each (obs, action) pair'''
        # An advantage estimator can use any number of steps into the future. From 1 to all of them:
        # A^1_t = -V(s_t) + r_t + gamma * V(s_{t+1})
        # A^2_t = -V(s_t) + r_t + gamma * r_{t+1} + gamma^2 * V(s_{t+2})
        # ...
        # A^inf_t = -V(s_t) + sum_{l=0}^inf gamma^l * r_{t + l}

        # We could really use any of these. A^inf is really just the discounted sum of rewards for the whole episode. Simple.

        # The Generalized Advantage Estimator (GAE) paper says that A^1 can be a biased estimator, while A^inf can have high variance.
        # So the GAE is just a weighted exponential average of A^1, A^2, ... , A^inf.

        # GAE = sum l=0^inf (gamma * lambda)^l * delta_{t+l}

        # Where
        # delta_t = A^1_t = -V(s_t} + r_t + gamma * V(s_{t+1})
        # delta_t = the TD(1) residual of V(s)

        observation_tensor = torch.stack(self.observations)
        value_net.eval()
        values = value_net(observation_tensor.float()) * reward_scalar + reward_offset
        value_net.train()

        deltas = np.zeros(len(self.observations))
        gae_estimates = np.zeros(len(self.observations))
        for i in range(0, len(self.observations)):
            if i == len(self.observations) - 1:
                deltas[i] = -values[i] + self.rewards[i]
            else:
                deltas[i] = -values[i] + self.rewards[i] + gamma * values[i+1]

        gae_estimates = discount_cumsum(deltas, gamma * lamda)

        gae_mean = np.mean(gae_estimates)
        gae_std = np.std(gae_estimates)
        gae_estimates = (gae_estimates - gae_mean) / gae_std

        return np.ndarray.tolist(gae_estimates)

    def __len__(self):
        return len(self.observations)

class MyNet(nn.Module):

    def __init__(self,input_size,output_size,output_distribution="categorical"):
        super(MyNet, self).__init__()
        self.output_distribution = output_distribution
        self.input_size = input_size
        self.output_size = output_size
        is_value_net = not(output_distribution == "categorical" or output_distribution == "normal")
        hidden_width = 64
        if (is_value_net):
            hidden_width = 256
        self.activation = nn.Tanh()
        self.fc0 = nn.Linear(input_size,hidden_width)
        self.dropout0 = nn.Dropout(DROPOUT_PROB)
        self.fc1 = nn.Linear(hidden_width,hidden_width)
        self.dropout1 = nn.Dropout(DROPOUT_PROB)
        # self.fc2 = nn.Linear(hidden_width,hidden_width)
        # self.dropout2 = nn.Dropout(DROPOUT_PROB)
        self.fc3 = nn.Linear(hidden_width,output_size)
        self.dropout3 = nn.Dropout(DROPOUT_PROB)

        if self.output_distribution == "normal":
            log_std = NORMAL_STARTING_LOG_STD * np.ones(output_size, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad = False)
        
    # I found that putting the BN directly after the FC (before the activation) results 
    # in smaller parameter gradients and losses (like reasonable)
    def forward(self,x):
        normalized_x = (x - observation_offset) / observation_scalar
        # print("x: ", x)
        # print("normalized x: ", normalized_x)
        hidden1 = self.dropout0(self.activation(self.fc0(normalized_x)))
        hidden2 = self.dropout1(self.activation(self.fc1(hidden1)))
        # hidden3 = self.dropout2(self.activation(self.fc2(hidden2)))
        out = self.dropout3(self.fc3(hidden2))
        if self.output_distribution == "categorical":
            return Categorical(logits=out)
        elif self.output_distribution == "normal":
            if USE_SQUASHING_FUNCTION:
                scaled_out = torch.tanh(out) * policy_scalar + policy_offset
            else:
                scaled_out = out * policy_scalar + policy_offset
            std = torch.exp(self.log_std)
            return Normal(scaled_out, std)
        else:
            return out.view(-1, self.output_size)

inputSize = env.observation_space.shape[0] # State

# This DNN outputs an action given a state.
if (isinstance(env.action_space, gym.spaces.Discrete)):
    policy_net = MyNet(inputSize, env.action_space.n, "categorical")
    policy_net_old = MyNet(inputSize, env.action_space.n, "categorical")
elif (isinstance(env.action_space, gym.spaces.Box)):
    policy_net = MyNet(inputSize, env.action_space.shape[0], "normal")
    policy_net_old = MyNet(inputSize, env.action_space.shape[0], "normal")

# This DNN outputs a value given a state
value_net = MyNet(inputSize, 1, "None")

def init_weights(net, gain):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            m.bias.data.fill_(0.0)

# init_weights(policy_net, 1e-2)
nn.init.orthogonal_(policy_net.fc3.weight, 1e-2)
policy_net.fc3.bias.data.fill_(0)
# init_weights(value_net, 1)

# policy_optimizer = torch.optim.Adam([p for p in policy_net.parameters() if p.requires_grad], POLICY_LR)
policy_optimizer = torch.optim.RMSprop([p for p in policy_net.parameters() if p.requires_grad], POLICY_LR)
policy_net_old.load_state_dict(policy_net.state_dict())
policy_net_old.eval()
policy_lr_scheduler = lr_scheduler.ExponentialLR(policy_optimizer, gamma=LR_GAMMA, verbose=True)

# value_optimizer = torch.optim.Adam([p for p in value_net.parameters() if p.requires_grad], VALUE_LR)
value_optimizer = torch.optim.RMSprop([p for p in value_net.parameters() if p.requires_grad], VALUE_LR)
value_lr_scheduler = lr_scheduler.ExponentialLR(value_optimizer, gamma=LR_GAMMA, verbose=True)

mean_episode_rewards_plot = []
policy_grad_loss_plot = []
value_loss_plot = []
policy_weights_plot = []
policy_biases_plot = []

def plot_durations():

    plt.figure(1)
    plt.clf()
    plt.title('Rewards')
    plt.xlabel('Epoch')
    plt.ylabel('Average reward')
    plt.plot(mean_episode_rewards_plot)

    # plt.figure(2)
    # plt.clf()
    # plt.title('Policy grad loss')
    # plt.xlabel('Batch')
    # plt.ylabel('Policy grad loss')
    # plt.plot(policy_grad_loss_plot)

    # plt.figure(3)
    # plt.clf()
    # plt.title('Value loss')
    # plt.xlabel('Batch')
    # plt.ylabel('Value loss')
    # plt.plot(value_loss_plot)

    plt.figure(4)
    plt.clf()
    plt.title('Policy Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Weights')
    plt.plot(policy_weights_plot)

    plt.figure(5)
    plt.clf()
    plt.title('Policy Biases')
    plt.xlabel('Epoch')
    plt.ylabel('Biases')
    plt.plot(policy_biases_plot)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    print("policy_net log std: ", policy_net_old.log_std)
    print("Reward scalar: ", reward_scalar, "    Reward offset: ", reward_offset)

def select_action(state, use_mean = False):
    state = state.reshape(-1,inputSize)
    with torch.no_grad():
        ans = policy_net_old(state)
        if use_mean:
            action = ans.mean
        else:
            action = ans.sample()
    # print("action mean: ", ans.mean)
    # print("action: ", action)
    # print("log prob: ", ans.log_prob(action))
    return action, ans.log_prob(action).sum(axis=-1)

def collect_experience(memories_list, index, epoch):

    env = gym.make(ENVIRONMENT)
    env.seed(seed + index + epoch)
        
    num_actions = 0
    while num_actions < NUM_ACTIONS_PER_SIM:
        memory = PPOEpisodeMemory()
        done = False
        state = torch.from_numpy(env.reset())
        while not done:
            # Select and perform an action
            action, log_prob_of_action = select_action(state.float())
            next_state, reward, done, _ = env.step(action.data.numpy().squeeze())

            # Save the experience
            memory.push(state, action, reward, done, log_prob_of_action)

            # Update the state
            state = torch.from_numpy(next_state)

            num_actions += 1

        memories_list.append(memory)
    env.close()
    

def run_eval_sim(env):
    eval_rewards = []
    
    # Initialize state and start simulating
    state = torch.from_numpy(env.reset())
    done = False
    while not done:
        # Select and perform an action
        action, log_prob_of_action = select_action(state.float(), use_mean = True)
        next_state, reward, done, _ = env.step(action.data.numpy().squeeze())
        state = torch.from_numpy(next_state)

        env.render()
        eval_rewards.append(reward)

    return np.sum(eval_rewards)


if __name__ == '__main__':
    for epoch in range(0, NUM_TRAINING_EPOCHS):
        # ------------------------------ Collect a batch of experience -------------------------------- #
        episode_memories = []  # <-- can be shared between processes.
        threads = []
        for i in range(NUM_PARALLEL_SIMS):
            t = threading.Thread(target=collect_experience, args=(episode_memories,i, epoch))  # Passing the list
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        discounted_cumulative_sums_of_rewards = []
        observations = []
        episode_rewards_plot = []
        for memory in episode_memories:
            discounted_cumulative_sums_of_rewards += np.ndarray.tolist(discount_cumsum(memory.rewards, GAMMA))
            observations += memory.observations
            episode_rewards_plot.append(np.sum(memory.rewards))
        discounted_cumulative_sums_of_rewards = torch.from_numpy(np.array(discounted_cumulative_sums_of_rewards))
        observations = torch.stack(observations)
        mean_episode_rewards_plot.append(np.mean(episode_rewards_plot))

        with torch.no_grad():
            observation_offset = observations.mean(axis=0).float()
            observation_scalar = observations.std(axis=0).float()
            observation_scalar[observation_scalar<1] = 1
            reward_scalar = torch.std(discounted_cumulative_sums_of_rewards)
            reward_offset = torch.mean(discounted_cumulative_sums_of_rewards)

        reward_scalar = torch.std(discounted_cumulative_sums_of_rewards)
        reward_offset = torch.mean(discounted_cumulative_sums_of_rewards)
        observations = []
        actions_taken = []
        advantages_of_actions_taken = []
        log_probs_of_actions_taken = []
        for memory in episode_memories:
            observations += memory.observations
            actions_taken += memory.actions
            log_probs_of_actions_taken += memory.log_probs_of_actions
            advantages_of_actions_taken += memory.get_generalized_advantage_estimates(GAMMA, LAMBDA, value_net)
        observations = torch.stack(observations)
        actions_taken = torch.stack(actions_taken).squeeze()
        log_probs_of_actions_taken = torch.stack(log_probs_of_actions_taken).squeeze()
        advantages_of_actions_taken = torch.from_numpy(np.array(advantages_of_actions_taken)) 
           
        # ------------------------------ Calculate the policy gradient -------------------------------- #
        # Vary the batch size
        NUM_OPT_STEPS = NUM_OPT_STEPS_PER_SAMPLE * (1 + int( discounted_cumulative_sums_of_rewards.size()[0] / BATCH_SIZE))
        print("number of opt steps", NUM_OPT_STEPS, "  batch size: ", BATCH_SIZE)
        # policy gradient = (gradient of log probability of policy choosing given action at given state w.r.t. theta) * (Advantage of given action at given state)
        indices = range(0, len(observations))
        skipped_opt_steps = 0
        for i in range(0, NUM_OPT_STEPS):
            sample_indices = np.random.choice(indices, BATCH_SIZE, replace=True)
            
            # zero_grad() zeros out the gradients of all of the parameters in a model.
            # Building the computational graph which will determine the gradients w.r.t. the parameters starts after zero_grad()
            policy_optimizer.zero_grad()
            new_log_probs_of_actions_taken = policy_net(observations[sample_indices].float()).log_prob(actions_taken[sample_indices])

            # If we have more than one action dimension, the probability of this action is each probability multiplied
            # Or it is the sum of the log probabilities
            if (len(new_log_probs_of_actions_taken.shape) > 1):
                new_log_probs_of_actions_taken = new_log_probs_of_actions_taken.sum(axis=-1)

            ratios = torch.exp(new_log_probs_of_actions_taken - log_probs_of_actions_taken[sample_indices])
            clipped_ratios = torch.clamp(ratios, 1 - PPO_EPSILON, 1 + PPO_EPSILON)
            negative_policy_grad = -torch.min(ratios * advantages_of_actions_taken[sample_indices], clipped_ratios * advantages_of_actions_taken[sample_indices]).mean()
            loss = negative_policy_grad
            loss.backward()

            grads = []
            for param in policy_net.parameters():
                try:
                    # param.grad.data.clamp_(-0.5, 0.5)
                    grads.extend(param.grad.data.abs().numpy().flatten())
                except:
                    pass

            # Approximate the KL divergence
            approx_kl_divergence = (new_log_probs_of_actions_taken - log_probs_of_actions_taken[sample_indices]).mean().item()
            # print("i: ", i, "  Approximate KL divergence: ", approx_kl_divergence)
            if (abs(approx_kl_divergence) > TARGET_KL):
                skipped_opt_steps += 1
            else:
                # Take the step
                policy_optimizer.step()

            # policy_grad_loss_plot.append(negative_policy_grad.detach().numpy())

            # print("log probs of actions taken: ", log_probs_of_actions_taken[sample_indices])
            # print("new log probs of actions taken: ", new_log_probs_of_actions_taken)

        print("Skipped ", skipped_opt_steps, " out of ", NUM_OPT_STEPS, " opt steps")
        policy_lr_scheduler.step()
        PPO_EPSILON = PPO_EPSILON * LR_GAMMA
        policy_net.log_std.data = policy_net.log_std.data - LOG_STD_DECREMENT_PER_EPOCH
        policy_net_old.log_std.data = policy_net_old.log_std.data - LOG_STD_DECREMENT_PER_EPOCH

        # Update policy_net_old
        policy_net_old.load_state_dict(policy_net.state_dict())

        for i in range(0, NUM_OPT_STEPS):

            sample_indices = np.random.choice(indices, BATCH_SIZE, replace=True)
             # -------------------- Update value function ----------------------- #
            value_optimizer.zero_grad()
            predicted_values_of_states = value_net(observations[sample_indices].float()) * reward_scalar + reward_offset
            value_loss = ((predicted_values_of_states.float() - discounted_cumulative_sums_of_rewards[sample_indices].view(-1,1).float())**2).mean()
            value_loss.backward()

            grads = []
            for param in value_net.parameters():
                try:
                    # param.grad.data.clamp_(-0.5, 0.5)
                    grads.extend(param.grad.data.abs().numpy().flatten())
                except:
                    pass

            # Take the step
            value_optimizer.step()

            # value_loss_plot.append(value_loss.detach().numpy())

        value_lr_scheduler.step()

        run_eval_sim(env)
        plot_durations()

        policy_weights = []
        policy_biases = []
        for m in policy_net.modules():
            if isinstance(m, nn.Linear):
                policy_weights += m.weight.detach().numpy().flatten().tolist()
                policy_biases += m.bias.detach().numpy().flatten().tolist()

        policy_weights_plot.append(np.mean(np.abs(policy_weights)))
        policy_biases_plot.append(np.mean(np.abs(policy_biases)))

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()  

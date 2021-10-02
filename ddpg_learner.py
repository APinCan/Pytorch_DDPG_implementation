# https://github.com/JL321/PolicyGradients-torch
# https://github.com/seungeunrho/minimalRL
# https://github.com/sfujim/TD3/blob/master/OurDDPG.py

import gym
from gym import ObservationWrapper
import mujoco_py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import copy
import os
import random
import logging
from logging.handlers import RotatingFileHandler
from collections import OrderedDict, deque


# ENV_NAME = 'HalfCheetah-v2'
ENV_NAME = 'Hopper-v2'
# ENV_NAME = 'InvertedPendulum-v2'
# ENV_NAME = 'Walker2d-v2'
# ENV_NAME = 'Reacher-v2'
ACTION_RANGE = 1.0
NUM_GPU = '1'
os.environ['CUDA_VISIBLE_DEVICES']=NUM_GPU
overall_dir_name="TensoboardWriter-"+ENV_NAME

device = torch.device('cuda') # cuda:0 ??

LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
TAU = 0.005
MINI_BATCH = 100
GAMMA = 0.99
BUFFER_SIZE = 1000000 # 10^6
EPISODES = 1000000
EPISODES_EVALUATION = 100
LIMITED_TIMESTEP = 1700000
# gaussian noise
MEAN = 0.0
STD = 0.1


def get_logger(log_dictname):
    logger = logging.getLogger('taxi_logger')
    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)sl%(filename)s:%(lineno)s] >> %(message)s')
    params_filename = log_dictname+'/hdqn-correctanswer2.log'

    file_handler=logging.handlers.RotatingFileHandler(filename=params_filename, maxBytes=50*1024*1024, backupCount=1000)

    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger


dir_path = "logs/"+overall_dir_name
os.makedirs(dir_path, exist_ok=True)
writer = SummaryWriter('runs/'+overall_dir_name)
logger = get_logger('logs/'+overall_dir_name)
logger.info("training start")


class FrameSkipEnv():
    def __init__(self, env, n_skip):
        self._env = env
        self._n_skip = n_skip
    
    def action_space(self):
        return self._env.action_space
    
    def step(self, action):
        acc_reward = 0
        for _ in range(self._n_skip):
            obs, reward, done, info = self._env.step(action)
            acc_reward += reward
            if done:
                break
                    
        return obs, acc_reward, done, info
    
    def reset(self):
        obs = self._env.reset()
        
        return obs
    
    def render(self):
        self._env.render()
        
    def close(self):
        self._env.close()

    
class Memory():
    def __init__(self):
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.memory_length = 0
        
    def put(self, transition):
        self.memory.append(transition)
        if self.memory_length < BUFFER_SIZE:
            self.memory_length += 1
         
    def sample(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        state_list, action_list, next_state_list, reward_list, done_list = [], [], [], [], []
        
        for transition in mini_batch:
            state, action, next_state, reward, done = transition
            reward_list.append(reward)
            done_list.append(done)
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
        
        return torch.tensor(state_list, dtype=torch.float, device=device), \
                torch.tensor(action_list, dtype=torch.float,  device=device), \
                torch.tensor(next_state_list, dtype=torch.float, device=device), \
                torch.tensor(reward_list, dtype=torch.float, device=device), \
                torch.tensor(done_list, device=device), \
                
    def size(self):
        return self.memory_length


class CriticBatchNorm(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(CriticBatchNorm, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions

        self.input_obs = nn.Linear(in_features=self.n_observations+self.n_actions, out_features=256) # new
        self.linear0 = nn.Linear(in_features=256, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=1)
        
    def forward(self, x, a, dim=0):
        data = torch.cat([x, a], dim=-1)
        data = F.relu(self.input_obs(data))
        data = F.relu(self.linear0(data))
        data = self.output(data)
        data = torch.squeeze(data, -1)
        
        return data
    

class ActorBatchNorm(nn.Module):
    def __init__(self, n_observations, n_actions, action_range=1.0):
        super(ActorBatchNorm, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.action_range = action_range
        
        self.input = nn.Linear(in_features=self.n_observations, out_features=256)
        self.linear0 = nn.Linear(in_features=256, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=self.n_actions)
        self.tanh = nn.Tanh()
        
    def forward(self, x, dim=0):
        x = x.view(-1, self.n_observations)
        data = F.relu(self.input(x))
        data = F.relu(self.linear0(data))
        data = self.output(data)
        data = self.tanh(data)
        data = torch.mul(data, self.action_range)
        
        return data
    
    
def training(critic, target_critic, actor, target_actor, mini_batch, actor_optimizer, critic_optimizer):
    state, action, next_state, reward, done = mini_batch
    
    # current q
    q = critic(state, action)
    
    # target q
    target_next_action = target_actor(next_state)
    target_q = target_critic(next_state, target_next_action) # (64,2)
    y = reward + GAMMA*target_q*done
    
    critic_loss = F.mse_loss(y.detach(), q)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    ##############################################
    action_grad = actor(state)
    qaction_grad = -critic(state, action_grad)
    actor_loss = qaction_grad.mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    return critic_loss, actor_loss
    

def update_targetnetwork(critic, target_critic, actor, target_actor, TAU):
    new_critic_params = OrderedDict()
    new_actor_params = OrderedDict()

    for layer_name in critic.state_dict():
        new_critic_params[layer_name] = critic.state_dict()[layer_name]*TAU + target_critic.state_dict()[layer_name]*(1-TAU)
        
    for layer_name in actor.state_dict():
        new_actor_params[layer_name] = actor.state_dict()[layer_name]*TAU + target_actor.state_dict()[layer_name]*(1-TAU)
        
    target_critic.load_state_dict(new_critic_params)
    target_actor.load_state_dict(new_actor_params)


env = gym.make(ENV_NAME)
n_observations = env.observation_space.shape[-1]
n_actions = env.action_space.shape[-1]

actor = ActorBatchNorm(n_observations, n_actions).cuda()
critic = CriticBatchNorm(n_observations, n_actions).cuda()
target_actor = ActorBatchNorm(n_observations, n_actions).cuda()
target_critic = CriticBatchNorm(n_observations, n_actions).cuda()
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR, weight_decay=0.005)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC, weight_decay=0.005)
writer.add_text('optimizer', 'actor critic both weight_decay=0.005')

replay_memory = Memory()
ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions))

t = 0
actor.train()
print("GPU:   "+NUM_GPU+"       "+overall_dir_name)
for epi in range(EPISODES):
    obs = env.reset()
    rewards = 0.0
    critic_loss = 0.0
    actor_loss = 0.0
    sub_t = copy.copy(t)
    done = False
    total_done = False
    
    while not done:
        action = actor(torch.tensor(obs, dtype=torch.float, device=device))
        action = action.detach().cpu().numpy()[0]
        action += np.random.normal(MEAN, STD*ACTION_RANGE, size=n_actions) # normal noise
        #################################################
        next_obs, reward, done, info = env.step(action) 
        rewards += reward
        #################################################
        replay_memory.put((obs, action, next_obs, reward, not done))
        
        if replay_memory.size() >= 25000:
            mini_batch = replay_memory.sample(MINI_BATCH)
            critic_loss, actor_loss = training(critic, target_critic, actor, target_actor, mini_batch, actor_optimizer, critic_optimizer)
            # target network update
            update_targetnetwork(critic, target_critic, actor, target_actor, TAU)

        t += 1
        obs = copy.copy(next_obs)
        # end
        if t>=LIMITED_TIMESTEP:
            total_done = True
            break
    
    writer.add_scalar('environment/timestep', (t-sub_t), t)
    writer.add_scalar('training/rewards', rewards, t)
    writer.add_scalar('training/actor_loss', actor_loss, t)
    writer.add_scalar('training/critic_loss', critic_loss, t)
    
    if total_done:
        break
    

torch.save(actor.state_dict(), './saved_models/actor_model-'+overall_dir_name+".pt")

actor.eval()
eval_env = gym.make(ENV_NAME)
print("evaluation phase")
# t = 0
for epi in range(EPISODES_EVALUATION):
    obs = eval_env.reset()
    rewards = 0.0
    sub_t = copy.copy(t)
    done = False
    rewards_list = []
    timestep_list = []
    
    while not done:
#         env.render()
        action = actor(torch.tensor(obs, dtype=torch.float, device=device))
        action = action.detach().cpu().numpy()[0]
        ########################################################
        next_obs, reward, done, info = eval_env.step(action)
        ########################################################
        
        obs = copy.copy(next_obs)
        rewards += reward
        t += 1
    
    rewards_list.append(rewards)
    timestep_list.append((t-sub_t))
    if epi%10==0:
        print("epi: "+str(epi)+"  reward mean: "+str(np.mean(rewards_list))+"  timestep: "+str(np.mean(timestep_list)))
        rewards_list = []
        timestep_list = []
        
    writer.add_scalar('evaluation/rewards', rewards, epi)
    
env.close()
eval_env.close()
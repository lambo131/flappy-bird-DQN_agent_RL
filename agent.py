import torch
import flappy_bird_gymnasium
import gymnasium
import itertools
import yaml # // for loading hyperparameters convieniently
import random
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime
import time
import math
import numpy as np
import msvcrt
import pickle
import argparse
import pandas as pd
import threading
import shutil

from dqn import DQN
from experience_replay import ReplayMemory

matplotlib.use("Agg")
DATE_FORMAT = "%m-%d %H:%M:%S"

# // create log folder
runs_dir = "./runs"
os.makedirs(runs_dir, exist_ok=True)

# specify to use gpu if avaiable as the computation device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device('cpu')  # // force to use CPU. Some times GPU overhead is larger than CPU overhead, especially for small networks
            
def get_current_time() -> str:
    return datetime.now().strftime(DATE_FORMAT)

def save_agent(agent, filepath):
    if hasattr(agent, 'lock'):
        del agent.lock     # Clean locks recursively
    with open(filepath, 'wb') as f:
        pickle.dump(agent, f)

def load_agent(filepath):
    with open(filepath, 'rb') as f:
        agent = pickle.load(f)
    return agent

def get_probability_mask(size):
    random_numbers = [random.random() for _ in range(size)]
    total = sum(random_numbers)
    return [x/total for x in random_numbers]

def get_values_sd_from_mean(input_list, sd):
    mean = np.mean(input_list)
    std_dev = np.std(input_list)
    if sd >= 0:
        threshold = mean + sd * std_dev
    else:
        threshold = mean - abs(sd) * std_dev
    values_sd_from_mean = [value for value in input_list if (sd >= 0 and value > threshold) or (sd < 0 and value < threshold)]
    
    return values_sd_from_mean

def get_env(agent, render: bool, audio_on=False, old_env=None):
    """
    Creates a new environment with specified render mode, closing the old one if it exists.
    
    Args:
        render (bool): Whether to render the environment
        old_env: The existing environment to close (optional)
        
    Returns:
        The newly created environment
    """
    # Close old environment if it exists
    if old_env is not None:
        old_env.close()
    
    current_render_mode = "human" if render else None
    
    # Create new environment based on env_id
    if agent.env_id == "CartPole-v1":
        env = gymnasium.make(
            agent.env_id, 
            render_mode=current_render_mode, 
            max_episode_steps=agent.episode_stop_reward
        )
    elif agent.env_id == "FlappyBird-v0":
        env = gymnasium.make(
            "FlappyBird-v0", 
            render_mode=current_render_mode, 
            use_lidar=False,
            audio_on=audio_on,
        )
    elif agent.env_id == "MountainCar-v0":
        env = gymnasium.make(
            agent.env_id, 
            render_mode=current_render_mode, 
            goal_velocity=0.1
        )
    elif agent.env_id == "LunarLander-v3":
        env = gymnasium.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    else:
        raise ValueError(f"Unknown environment ID: {agent.env_id}")
    
    return env


class Agent:
    # // the Agent __init__ function defines the hyperparaters of the agent
    def __init__(self, hyperparameter_set, log=True):
        # settings from hyperparameters.yml
        yaml_path = "./hyperparameters.yml"
        with open(yaml_path, 'r') as file:
            yaml_all = yaml.safe_load(file)
            hyperparameter = yaml_all[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set
        self.env_id = hyperparameter['env_id']
        self.train_stop_reward = hyperparameter['train_stop_reward']
        self.episode_stop_reward = hyperparameter['episode_stop_reward']
        self.replay_memory_size = hyperparameter['replay_memory_size']
        self.batch_size = hyperparameter['mini_batch_size']
        self.initial_epsilon = hyperparameter['initial_epsilon']
        self.min_epsilon = hyperparameter['min_epsilon']
        self.epsilon_decay = hyperparameter['epsilon_decay']
        self.discount_factor = hyperparameter['discount_factor']
        self.target_sync_freq = hyperparameter['target_sync_freq']
        # // Initialze policy and target network, and send the network instance to GPU
        self.hidden_nodes = hyperparameter['hidden_nodes']
        self.hidden_layers = hyperparameter['hidden_layers']
        # Initialize DQN network ---------
        self.policy_network = DQN(hyperparameter['state_dim'], hyperparameter['action_dim'], self.hidden_nodes, self.hidden_layers).to(device=device)
        self.target_network = DQN(hyperparameter['state_dim'], hyperparameter['action_dim'], self.hidden_nodes, self.hidden_layers).to(device=device)
        # // MLP settings
        self.loss_fn = torch.nn.MSELoss()  # // loss function for training the DQN network
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=hyperparameter['learning_rate'])

        # -------------------class variables-------------------
        self.enable_log = log
        # // run folder files
        self.LOG_FILE = os.path.join(runs_dir, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(runs_dir, f'{hyperparameter_set}.pt')
        self.OLD_MODEL_FILE = os.path.join(runs_dir, f'{hyperparameter_set}_old.pt')
        self.GRAPH_FILE = os.path.join(runs_dir, f'{hyperparameter_set}.png')
        self.EXPLORATION_FILE = os.path.join(runs_dir, f'{hyperparameter_set}.csv')
        self.EXPERIENCE_FILE = os.path.join(runs_dir, f'{hyperparameter_set}_exp.pkl')
        # // memory
        self.best_reward = -float('inf')
        self.experience = ReplayMemory(self.replay_memory_size)
        self.important_experience = ReplayMemory(self.replay_memory_size)
        self.exploration_df = pd.DataFrame(columns=['episode', 'epsilon', 'reward'])    
        self.epsilon = self.initial_epsilon

        self.save_old_network()

        self.to_log(f"\n{get_current_time()}: creating agent...",print_enable=True)
        self.to_log(f"GPU Name: {torch.cuda.get_device_name(device)}", print_enable=True) 
        self.to_log("", print_enable=True)

    def __del__(self):
        print(f"Destructor called for {self.hyperparameter_set}")

    def to_log(self, str="", print_enable=True):
        if print_enable:
            print(str)
        if self.enable_log == True:
            with open(self.LOG_FILE, 'a') as file:
                file.write(str + '\n')

    def save_experience(self, file_path):
        '''memory_temp = ReplayMemory(self.experience.__len__()+self.important_experience.__len__())
        memory_temp.merge_memory(self.experience)
        memory_temp.merge_memory(self.important_experience)'''
        all_experience = (self.experience, self.important_experience)

        with open(file_path, 'wb') as f:
            pickle.dump(all_experience, f)
        
        self.to_log(f"saving all experience to: {file_path}", print_enable=True)
        self.to_log(f"\texp len: {all_experience[0].__len__()}, **exp len: {all_experience[1].__len__()}\n", print_enable=True)

    def load_experience(self, file_path=None):
        if file_path is None:
            file_path = self.EXPERIENCE_FILE
        if os.path.exists(file_path):
            self.to_log(f">>> Loading experience from {file_path}", print_enable=True)
            temp_memory = pickle.load(open(file_path, 'rb'))
            self.experience = temp_memory[0]
            self.important_experience = temp_memory[1]
            self.to_log(f"current exp len: {self.experience.__len__()}, **exp len: {self.important_experience.__len__()}\n", print_enable=True)
        else:  
            print(f"### could not load experience, {file_path} not found.\n")
    
    def save_old_network(self):
        if os.path.exists(self.MODEL_FILE):
            shutil.copyfile(self.MODEL_FILE, self.OLD_MODEL_FILE)

    def save_policy_network(self, model_path):
        print(f">>> saving policy network to {model_path}\n")
        torch.save(self.policy_network.state_dict(), model_path)

    def load_policy_network(self, model_path=None):
        if model_path is None:
            model_path = self.MODEL_FILE
        if os.path.exists(model_path):
            print(f">>> Loading policy network from {model_path}\n")
            self.policy_network.load_state_dict(torch.load(model_path))
        else:  
            print(f"### could not load policy network, {model_path} not found.\n")

    def get_experience(self):
        return self.experience

    def get_important_experience(self):
        return self.important_experience

    def forget_experience(self, percentage=1):
        self.experience.clear(percentage) 
        self.important_experience.clear(percentage)

    def get_exploration_prob_mask(self, action_dim):
        action_0_prob = 0.8 - 0.2 + random.random() * 0.4  # no flap action probability distribution from 0.6 to 1.0
        return np.array([action_0_prob, 1-action_0_prob]) # [0.6,0.4] ~ [1.0, 0]
                         
    def run(self, episodes=-1, render=False):
        self.to_log(f">>> creating {self.env_id} for *running* the agent...\n", print_enable=True)

        env = get_env(self, render=False)

        count_transitions = 0
        current_render_en = render
        audio_en = False
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # // episode loop:----------------
        for episode in itertools.count():
            if episodes >=1 and episode > episodes:
                break

            # toggle render
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'r':  # ESC key
                    current_render_en = not current_render_en
                    audio_en = not audio_en
                    env = get_env(self, render=not current_render_en,audio_on=not audio_en, old_env=env)      

            #  reset the environment for a new episode
            obs, _ = env.reset()
            episode_reward = 0.0 # initalize episode reward

            terminated, truncated = (False, False)
            while (not terminated) and (not truncated):
                # Next action:
                # // only using policy network for inference, disable gradient calculation in pytorch
                with torch.no_grad(): 
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    action = self.policy_network(obs_tensor.unsqueeze(0)).squeeze().argmax().item()
            
                # -----------------step environment-----------------
                # use .item() to convent simple 1-element tensor to a simple number
                new_obs, reward, terminated, _, info = env.step(action) 
                count_transitions+=1

                # -----------------Processing:------------------
                # // update episode reward
                episode_reward += reward

                # move to new state(obs)
                obs = new_obs
                
            
            print(f"Ep: {episode}, trans: {count_transitions}, R: {episode_reward:.1f}")

    def explore(self, render=False, epsilon=0.01, episodes=-1, recursive=False, depth=0):
        self.to_log(f">>> creating {self.env_id} for *exploring* the agent...\n", print_enable=True)

        env = get_env(self, render=False)

        count_transitions = 0
        reward_history = []
        epsilon_history = []
        prob_mask_history = []

        current_render_en = render
        # epsilon = 0.01

        # // episode loop:----------------
        for episode in itertools.count():
            if episodes >=1 and episode > episodes:
                break

            # toggle render
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'r':  # ESC key
                    current_render_en = not current_render_en
                    env = get_env(self, render=not current_render_en, old_env=env)      

            #  reset the environment for a new episode
            obs, _ = env.reset()
            episode_reward = 0.0 # initalize episode reward
            # random epsilon
            epsilon = (epsilon*0.5) + random.random()*epsilon
            epsilon_history.append(epsilon)
            # random probability mask
            exploration_prob_mask = self.get_exploration_prob_mask(action_dim=env.action_space.n)
            prob_mask_history.append(exploration_prob_mask)

            terminated, truncated = (False, False)
            while (not terminated) and (not truncated):
                # Next action:
                if random.random() < epsilon:
                    action = env.action_space.sample(probability=exploration_prob_mask)
                else:
                    # // only using policy network for inference, disable gradient calculation in pytorch
                    with torch.no_grad(): 
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                        action = self.policy_network(obs_tensor.unsqueeze(0)).squeeze().argmax().item()
            
                # -----------------step environment-----------------
                # use .item() to convent simple 1-element tensor to a simple number
                new_obs, reward, terminated, _, info = env.step(action) 
                count_transitions+=1

                # -----------------Processing:------------------
                self.experience.append((obs, action, new_obs, reward, terminated))   # ********************
                # // update episode reward
                episode_reward += reward

                # move to new state(obs)
                obs = new_obs
                
            reward_history.append(episode_reward)

            print(f"Ep: {episode}, trans: {count_transitions}, epsilon: {epsilon:.5f}, " +\
                  f"R: {episode_reward:.1f}, mem_len: {self.experience.__len__()}")
            
        self.to_log(f">>> exploration completed: "+\
              f"\tmean reward: {np.mean(reward_history)}, best reward: {np.max(reward_history)}", print_enable=True)
        print("\t+1sd reward:\n\t", end="")
        for num in get_values_sd_from_mean(reward_history, 1):
            print(f"{num:.2f}, ", end="")
        print()

        if np.max(reward_history) > self.best_reward:
            self.best_reward = np.max(reward_history)
            temp_best_reward = self.best_reward
            if recursive:
                self.to_log(f"\tstarting recursive explore: {self.best_reward}", print_enable=True)
                self.explore(episodes=100,depth=depth+1)
                self.best_reward = temp_best_reward
            if depth == 0:
                self.to_log(f"\tnew best reward(explore): {self.best_reward}", print_enable=True)

        # Create a Pandas DataFrame from the dictionary
        exploration_data = {'epsilon': epsilon_history, 'prob_mask': prob_mask_history, 'reward': reward_history}
        self.exploration_df = pd.DataFrame(exploration_data)
        self.exploration_df.to_csv(self.EXPLORATION_FILE, index=True)
        

    def train(self, render=False, episodes=99999):

        self.to_log(f"creating {self.env_id} for *training* the agent...\n", print_enable=True)

        # // create the environment
        env = get_env(self, render=render)
        # copy the policy network to the target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        # // initialize Replay Memory
        memory = self.get_experience()

        # // variables
        rewards_per_episode = []
        best_reward = -float('inf')
        epsilon_history = []
        total_transitions = 0 # for logging and debugging
        count_transitions = 0 # // for target network sync frequency
        start_time = datetime.now()
        current_render_en = render
         # initialize epsilon
        epsilon_normal = self.initial_epsilon
        # ---- experimentation with dynamic epsilon -----
        exploration_mode_epsilon = 0.5
        avg_episode_R = 0
        avg_episode_R_alpha = 0.9
        # ---- experimentation with dynamic replay memory
        last_mem_len = 0
        

        # // episode loop:----------------
        self.to_log(f"{start_time.strftime(DATE_FORMAT)}: Training starting...\n", print_enable=True)
       
        for episode in itertools.count():
            # toggle render
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'r':  # r key
                    current_render_en = not current_render_en
                    env = get_env(self, render=not current_render_en, old_env=env)               

            #  reset the environment for a new episode
            obs, _ = env.reset()
            exploration_prob_mask = self.get_exploration_prob_mask(action_dim=env.action_space.n)
            episode_reward = 0.0 # initalize episode reward
            reward_sd = 0
            skip_train = False
            # // epsilon experiment -------
            epsilon = epsilon_normal
            transitions_per_episode = 0
            episode_memory_appended = 0
            # // experiment with important experience
            episode_transitions = []

            terminated, truncated = (False, False)
            while (not terminated) and (not truncated) and episode_reward < self.episode_stop_reward:                
                # Next action:
                # (feed the observation to your agent here)
                if random.random() < epsilon:
                    action = env.action_space.sample(probability=exploration_prob_mask)
                else:
                    # // only using policy network for inference, disable gradient calculation in pytorch
                    with torch.no_grad(): 
                        # // pytoch networks expects input as batch dimension shape
                        # now, obs_tensor looks like this: tensor([1,2, ..])
                        # but we want the input to look like this, even if the number of input in a batch is 1:
                        #  -> tensor([[1,2, ..],[..]])

                        # unsqueeze(0) adds a new dimension at the 0th index, making it a batch of size 1
                        # the squeeze method turns the network output into a 1D tensor for the argmax operation
                        # ### note that the argmax() function here is a pytorch function, that returns a tensor
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                        action = self.policy_network(obs_tensor.unsqueeze(0)).squeeze().argmax().item()
                
                # -----------------step environment-----------------
                # use .item() to convent simple 1-element tensor to a simple number
                new_obs, reward, terminated, truncated, info = env.step(action) 

                # -----------------Processing:------------------
                if len(rewards_per_episode) >= 50:
                    reward_sd = np.std(random.sample(rewards_per_episode, 50))
                total_transitions+=1
                transitions_per_episode+=1
                # // update episode reward
                episode_reward += reward
                # // add experience (transition) as a tuple to Replay Memory
                memory.append((obs, action, new_obs, reward, terminated))
                episode_transitions.append((obs, action, new_obs, reward, terminated))
                
                # -------change epsilon dynamically during training-------
                # gradually decay epsilon
                epsilon_normal = max(self.min_epsilon, epsilon_normal * self.epsilon_decay)
                
                if random.random() < 0.5 and epsilon_normal < self.min_epsilon*1.01:
                    epsilon_normal = 0.3
                epsilon = epsilon_normal
                # enable exploration mode when the episode reward is high enough
                '''
                if episode_reward - avg_episode_R > 2 * reward_sd:
                    epsilon = exploration_mode_epsilon
                '''
                # move to new state(obs)
                obs = new_obs

            # ^^^^^^ end of an episode --------------------------------------------------------------------------------------
            # print(">>> episode completed...")

            rewards_per_episode.append(episode_reward)
            epsilon_history.append(epsilon)
            if len(rewards_per_episode) % 50 == 0:
                self.save_graph(rewards_per_episode, epsilon_history)

            # --- experimentation with dynamic epsilon ---
            # calculate avg reward per episode
            avg_episode_R = avg_episode_R_alpha * avg_episode_R + (1 - avg_episode_R_alpha) * episode_reward

            # --------------save best policy network weights----------------
            # forget some of the older transitions in the replay memory if reward is higher than last best reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.to_log(f"{datetime.now().strftime(DATE_FORMAT)}: New best reward: {self.best_reward} at episode {episode}. Model saved.\n", print_enable=True)
                
                last_mem_len = memory.__len__()
                # save the policy network if the episode reward is better than the best reward
                self.save_policy_network(self.MODEL_FILE)
                # ---- experimentation with dynamic replay memory ----
                # experiment with important experience
                for transition in episode_transitions:
                    self.important_experience.append(transition)
                self.save_experience(self.EXPERIENCE_FILE)
                # forget some of the older transitions
                memory.clear(0.5)
                #skip_train = True
                temp_best_reward = self.best_reward
                self.explore(episodes=100, epsilon=0.0005)  # to explore world with this good model
                self.best_reward = temp_best_reward # do not let explore update best reward here
            # --------------network update----------------
            for i in range(transitions_per_episode):
                # // check if the replay memory is large enough to sample a mini-batch
                # Also, after dynamic memory clearing, wait for memory to fill up again
                if skip_train:
                    self.to_log(f"skipping training...\n")
                    break
                if memory.__len__() < max(self.batch_size*1, last_mem_len):
                    break
              
                mini_batch_1 = memory.sample(self.batch_size) # a mini-batch of transitions for training
                if self.important_experience.__len__() >= self.batch_size:
                    mini_batch_2 = self.important_experience.sample(int(self.batch_size * 0.2))
                    train_mini_batch = tuple(torch.concatenate((a, b)) for a, b in zip(mini_batch_1, mini_batch_2))
                else:
                    train_mini_batch = mini_batch_1
                # ******call optimize function (training the DQN network)*******
                # self.optimize_v2(mini_batch, self.policy_network, self.target_network)
                self.optimize_double_dqn(train_mini_batch, self.policy_network, self.target_network)
                # // sync target network with policy network
                if count_transitions > self.target_sync_freq:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                    count_transitions = 0
                    # print("sync target network with policy network...")

                count_transitions+=1
            # ------------------------------------------------------------------------
                
            print(f"Ep: {episode}, Trans#: {total_transitions}, R: {episode_reward:.1f}, avg_ep_R: {avg_episode_R:.2f}, " +\
                  f"epsilon: {epsilon:.5f}, exploration_prob: {exploration_prob_mask}，" +\
                  f"mem_len: {memory.__len__()}, **men_len: {self.important_experience.__len__()}, ")
            
            # // breaking training loop if the average reward is high enough
            if avg_episode_R >= self.train_stop_reward or episode >= episodes:
                self.to_log(f"\nTraining stoped after {episode} episodes with average reward: {avg_episode_R:.2f}\n", print_enable=True)
                break
        
        # self.key_listner.stop()
    
    def wrap_up(self):
        self.to_log(f"\n>>> Wrapping up...\n", print_enable=True)
        self.save_experience(self.EXPERIENCE_FILE)
        self.to_log(f"\n>>> agent exit...\n", print_enable=True)

    def save_graph(self, rewards_per_episode, epsilon_history):
       # plots two subplots: one for rewards per episode and one for epsilon history
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(rewards_per_episode, label='Rewards per episode')
        ax1.set_title('Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax2.plot(epsilon_history, label='Epsilon History', color='orange')
        ax2.set_title('Epsilon History')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')   
        ax2.legend()
        # save the graph to a file
        plt.savefig(self.GRAPH_FILE)
        plt.close(fig)  # close the figure to free memory

    def optimize_double_dqn(self, mini_batch, policy_network, target_network):
        obs_batch, action_batch, new_obs_batch, reward_batch, terminated_batch = mini_batch

        # // calculate Q values
        current_batch_q_value = policy_network(obs_batch).gather(dim=1, index=action_batch.unsqueeze(1)).squeeze()

        # // only the policy network is trained, so we disable gradient calculation for the target network
        with torch.no_grad():
            # // gets the q values in a batch, based on the ation taken
            # // the gather function gathers values along an axis specified by the index tensor
            # // the dim=1 means we are gathering values along the action dimension
            # // double DQN, uses the policy network to select actions and the target network to evaluate them
            next_q_values = policy_network(new_obs_batch)
            next_actions = next_q_values.argmax(dim=1)
            target_q_values = reward_batch + (1 - terminated_batch) * self.discount_factor * next_q_values.gather(dim=1, index=next_actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_batch_q_value, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def optimize_v2(self, mini_batch, policy_network, target_network):
        obs_batch, action_batch, new_obs_batch, reward_batch, terminated_batch = mini_batch

        # // calculate Q values
        current_batch_q_value = policy_network(obs_batch).gather(dim=1, index=action_batch.unsqueeze(1)).squeeze()
        # // only the policy network is trained, so we disable gradient calculation for the target network
        with torch.no_grad():
            # // gets the q values in a batch, based on the ation taken
            # // the gather function gathers values along an axis specified by the index tensor
            # // the dim=1 means we are gathering values along the action dimension
            next_q_values = target_network(new_obs_batch)
            # // calculate target Q values
            # // a neat way to include terminated state -> target = reward
            # .max(dim=1) returns max value in the action dimension, where as dim=0 is the batch dimension
            target_q_values = reward_batch + (1 - terminated_batch) * self.discount_factor * next_q_values.max(dim=1)[0]
        
        loss = self.loss_fn(current_batch_q_value, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def optimize_v1(self, mini_batch, policy_network, target_network):
        # this version of the optimize function is a more basic version, without batch processing
        # it is slower than the optimize_v2 function
        for obs, action, new_obs, reward, terminated in mini_batch:
            # // calculate Q values
            with torch.no_grad():
                q_values = policy_network(obs.unsqueeze(0))
                next_q_values = target_network(new_obs.unsqueeze(0))

            # // calculate target Q value
            # // a neat way to include terminted state -> target = reward
            target_q_value = reward + (1 - terminated) * self.discount_factor * next_q_values.max()

            loss = self.loss_fn(q_values[0], target_q_value)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train or run DQN agent')
    parser.add_argument('hyperparameter_set', type=str, help='name of RL env')
    parser.add_argument('model_name', type=str, default='', help='name of the model')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--run', action='store_true', help='Run the agent')
    parser.add_argument("--render", action='store_true', help='Render the environment during training or running')
    parser.add_argument("--load_exp", action='store_true', help='Load experience from file')
    parser.add_argument("--load_policy", action='store_true', help='Load policy network from file')
    args = parser.parse_args()

    # create DQN agent
    my_agent = Agent(hyperparameter_set=args.hyperparameter_set)
    
    if args.train:
        # Train the agent
        if args.load_exp:
            my_agent.load_experience(file_path=my_agent.EXPERIENCE_FILE)
        if args.load_policy:
            my_agent.load_policy_network()
            my_agent.explore(episodes=100, epsilon=0.0005)
        
        my_agent.train(render=args.render)
        # Save the trained agent to a file
        # save_agent(my_agent, f'{runs_dir}\{args.hyperparameter_set}_{args.model_name}.pkl')
    
    elif args.run:
        # Load the trained agent from a file
        if args.load_policy:
            my_agent.load_policy_network()
        # Run the agent in the environment
        my_agent.run()

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

from dqn import DQN
from experience_replay import ReplayMemory

DATE_FORMAT = "%m-%d %H:%M:%S"

# // create log folder
runs_dir = "C:\\Users\\ASUS\\OneDrive - HKUST Connect\\1_Projects\\FYP\\Reinforcement learning\\Flappy bird DQN project\\runs"
os.makedirs(runs_dir, exist_ok=True)

matplotlib.use("Agg")

# specify to use gpu if avaiable as the computation device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device('cpu')  # // force to use CPU. Some times GPU overhead is larger than CPU overhead, especially for small networks

def load_agent(filepath):
    with open(filepath, 'rb') as f:
        agent = pickle.load(f)
    return agent

class Agent:
    # // the Agent __init__ function defines the hyperparaters of the agent
    def __init__(self, hyperparameter_set):
        yaml_path = "C:\\Users\\ASUS\\OneDrive - HKUST Connect\\1_Projects\\FYP\\Reinforcement learning\\Flappy bird DQN project\\hyperparameters.yml"
        with open(yaml_path, 'r') as file:
            yaml_all = yaml.safe_load(file)
            hyperparameter = yaml_all[hyperparameter_set]

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
        # Initialize DQN network ---------
        self.policy_network = DQN(hyperparameter['state_dim'], hyperparameter['action_dim'], self.hidden_nodes).to(device=device)
        self.target_network = DQN(hyperparameter['state_dim'], hyperparameter['action_dim'], self.hidden_nodes).to(device=device)
        # // MLP settings
        self.loss_fn = torch.nn.MSELoss()  # // loss function for training the DQN network
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=hyperparameter['learning_rate'])

        # // run folder files
        self.LOG_FILE = os.path.join(runs_dir, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(runs_dir, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(runs_dir, f'{hyperparameter_set}.png')

    def to_log(self, str="", print_enable=False):
        if print_enable:
            print(str)

        with open(self.LOG_FILE, 'a') as file:
            file.write(str + '\n')

    def load_policy_network(self, model_path=None):
        if model_path is None:
            model_path = self.MODEL_FILE
        if os.path.exists(model_path):
            print(f">>> Loading policy network from {model_path}\n")
            self.policy_network.load_state_dict(torch.load(model_path))
        else:  
            print(f"### could not load policy network, {model_path} not found.\n")


    def run(self):
        self.to_log(f">>> creating environment {self.env_id} for running the agent...\n", print_enable=True)
        env = gymnasium.make(self.env_id, render_mode="human")

        count_transitions = 0
        # Initialize DQN network ---------
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # // episode loop:----------------
        for episode in itertools.count():
            
            #  reset the environment for a new episode
            obs, _ = env.reset()
            episode_reward = 0.0 # initalize episode reward

            terminated = False
            while not terminated:
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


    def train(self, render=False, episodes=100):
        self.to_log(f"creating environment {self.env_id} for running the agent...\n", print_enable=True)
        env = gymnasium.make(self.env_id, render_mode="human" if render else None)

        # // run() variables-----
        rewards_per_episode = []
        best_reward = -float('inf')
        epsilon_history = []
        total_transitions = 0 # for logging and debugging
        count_transitions = 0 # // for target network sync frequency
        start_time = datetime.now()
         # initialize epsilon
        epsilon_normal = self.initial_epsilon
        # ---- experimentation with dynamic epsilon -----
        exploration_mode_epsilon = 0.5
        avg_episode_R = 0.0
        avg_episode_R_alpha = 0.8
        # // initialize Replay Memory
        memory = ReplayMemory(self.replay_memory_size)
        # copy the policy network to the target network
        self.target_network.load_state_dict(self.policy_network.state_dict())


        # // episode loop:----------------
        self.to_log(f"{start_time.strftime(DATE_FORMAT)}: Training starting...\n", print_enable=True)
        for episode in itertools.count():
            
            #  reset the environment for a new episode
            obs, _ = env.reset()
            episode_reward = 0.0 # initalize episode reward
            # // epsilon experiment -------
            epsilon = epsilon_normal
            episode_transitions = 0

            terminated = False

            while not terminated and episode_reward < self.episode_stop_reward:
                # Next action:
                # (feed the observation to your agent here)
                if random.random() < epsilon:
                    action = env.action_space.sample()
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
                new_obs, reward, terminated, _, info = env.step(action) 
                total_transitions+=1
                episode_transitions+=1

                # -----------------Processing:------------------
                # // update episode reward
                episode_reward += reward
                # // add experience (transition) as a tuple to Replay Memory
                memory.append((obs, action, new_obs, reward, terminated))

                # -------change epsilon dynamically during training-------
                # gradually decay epsilon
                epsilon_normal = max(self.min_epsilon, epsilon_normal * self.epsilon_decay)
                # enable exploration mode when the average reward is higher than the mean episode reward
                if episode_reward > avg_episode_R*1.2:
                    epsilon = exploration_mode_epsilon
                else:
                    epsilon = epsilon_normal

                # move to new state(obs)
                obs = new_obs

            # ^^^^^^ end of an episode ---------------------------
            print(">>> episode completed...")

            rewards_per_episode.append(episode_reward)
            epsilon_history.append(epsilon)
            if len(rewards_per_episode) % 100 == 0:
                self.save_graph(rewards_per_episode, epsilon_history)

            # --- experimentation with dynamic epsilon ---
            # calculate avg reward per episode
            avg_episode_R = avg_episode_R_alpha * avg_episode_R + (1 - avg_episode_R_alpha) * episode_reward

            # --------------network update----------------
            # forget some of the older transitions in the replay memory if reward is higher than last best reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                # save the policy network if the episode reward is better than the best reward
                torch.save(self.policy_network.state_dict(), self.MODEL_FILE)
                self.to_log(f"{datetime.now().strftime(DATE_FORMAT)}: New best reward: {best_reward} at episode {episode}. Model saved.", print_enable=True)
                memory.clear(0.9)
                continue  # // run the next episode without training the network if episode is good

            #****** train policy network when the replay memory is large enough*******
            for i in range(episode_transitions):
                if memory.__len__() < self.batch_size:
                    break
              
                mini_batch = memory.sample(self.batch_size) # a mini-batch of transitions for training
                # ******call optimize function (training the DQN network)*******
                self.optimize_v2(mini_batch, self.policy_network, self.target_network)

                # // sync target network with policy network
                if count_transitions > self.target_sync_freq:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                    count_transitions = 0
                    #print("sync target network with policy network...")

                count_transitions+=1
            print(f"ep_trans:{episode_transitions}, mem_len: {memory.__len__()}, ")
            # ------------------------------------------------------------------------
                

            # // breaking training loop if the average reward is high enough
            if avg_episode_R >= self.train_stop_reward or sum(rewards_per_episode) > 500000:
                self.to_log(f"\nTraining stoped after {episode} episodes with average reward: {avg_episode_R:.2f}\n", print_enable=True)
                break

            print(f"Ep: {episode}, Trans#: {total_transitions}, R: {episode_reward:.1f}, epsilon: {epsilon:.2f}, eps-norm: {epsilon_normal:.2f}, avg_ep_R: {avg_episode_R:.2f}")

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

import pickle
import argparse

# Function to save the agent object
def save_agent(agent, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(agent, f)


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train or run DQN agent')
    parser.add_argument('hyperparameter_set', type=str, help='name of RL env')
    parser.add_argument('model_name', type=str, default='', help='name of the model')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--run', action='store_true', help='Run the agent')
    args = parser.parse_args()

    my_agent = Agent(hyperparameter_set=args.hyperparameter_set)
    if args.train:
        # Train the agent
        my_agent.train(render=False, episodes=1000)
        # Save the trained agent to a file
        my_agent.to_log(f"\n{datetime.now().strftime(DATE_FORMAT)}: Training completed. Saving agent with pickle...\n")
        save_agent(my_agent, f'{runs_dir}\{args.hyperparameter_set}_{args.model_name}.pkl')
    elif args.run:
        # Load the trained agent from a file
        my_agent.load_policy_network()
        # Run the agent in the environment
        my_agent.run()

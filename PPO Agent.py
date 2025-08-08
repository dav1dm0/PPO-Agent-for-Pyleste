import numpy as np  # For numerical operations
import torch  # For tensor operations
import torch.nn as nn  # For building neural networks
import torch.optim as optim  # For optimization algorithms
from torch.distributions import Categorical  # For categorical action sampling
from PICO8 import PICO8  # Pyleste emulator
from Carts.Celeste import Celeste  # Celeste game cart
import CelesteUtils as utils  # Utility functions for Celeste
import time
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99  # Discount factor for future rewards
LR = 3e-4  # Learning rate for optimizer
EPSILON = 0.2  # Clipping parameter for PPO
BATCH_SIZE = 64  # Batch size for updates
EPOCHS = 10  # Number of epochs for policy updates

# Actor-Critic Neural Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),  # Shared layer for both actor and critic
            nn.ReLU()  # Activation function
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),  # Output layer for action probabilities
            nn.Softmax(dim=-1)  # Softmax activation for probabilities
        )
        self.critic = nn.Linear(128, 1)  # Output layer for state value

    def forward(self, state):
        common = self.common(state)  # Shared computation
        return self.actor(common), self.critic(common)  # Actor and critic outputs

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)  # Policy network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)  # Optimizer
        self.memory = []  # Memory to store transitions

    def act(self, state):
        state = torch.FloatTensor(state).to(device)  # Convert state to tensor
        probs, _ = self.policy(state)  # Get action probabilities
        dist = Categorical(probs)  # Create a categorical distribution
        action = dist.sample()  # Sample an action

        # Map action to Pyleste commands
        action_mapping = {
            1: 0b000001,  # Press Left
            2: 0b000010,  # Press Right
            3: 0b010000,  # Press Jump
            4: 0b010001,  # Press Left + Jump
            5: 0b010010,  # Press Right + Jump
            #6: 0b100001,  # Press Left + Dash
            #7: 0b100010,  # Press Right + Dash
            6: 0b100100,  # Press Up + Dash
            7: 0b100101,  # Press Left + Up + Dash
            8: 0b100110,  # Press Right + Up + Dash
        }
        btn_state = action_mapping.get(action.item(), 0)  # Default to no action
        p8.set_btn_state(btn_state)  # Set button state

        p8.step()  # Advance the game by one frame

        return action.item(), dist.log_prob(action)  # Return action and log probability

    def store_transition(self, transition):
        self.memory.append(transition)  # Store transition in memory

    def compute_advantages(self, rewards, dones, values):
        advantages, returns = [], []  # Initialize advantages and returns
        G = 0  # Initialize return
        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
            G = reward + GAMMA * G * (1 - done)  # Compute return
            returns.insert(0, G)  # Store return
            advantages.insert(0, G - value)  # Compute advantage
        return torch.tensor(advantages).to(device), torch.tensor(returns).to(device)  # Return tensors

    def update_policy(self):
        states, actions, log_probs, rewards, dones, values = zip(*self.memory)  # Unpack memory
        advantages, returns = self.compute_advantages(rewards, dones, values)  # Compute advantages

        states = torch.FloatTensor(states).to(device)  # Convert states to tensor
        actions = torch.tensor(actions).to(device)  # Convert actions to tensor
        old_log_probs = torch.stack(log_probs).detach().to(device)  # Detach log probabilities

        for _ in range(EPOCHS):
            # Recompute policy outputs to rebuild the computational graph
            probs, values = self.policy(states)  # Get new probabilities and values
            dist = Categorical(probs)  # Create new categorical distribution
            new_log_probs = dist.log_prob(actions)  # Compute new log probabilities

            # Compute the ratio and advantages (detached for safety)
            ratio = (new_log_probs - old_log_probs).exp()
            surrogate1 = ratio * advantages.detach()
            surrogate2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages.detach()

            # Compute actor and critic losses
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = ((returns.detach() - values) ** 2).mean()

            loss = actor_loss + 0.5 * critic_loss  # Total loss

            # Backpropagate and update policy
            self.optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Backpropagate loss
            self.optimizer.step()  # Update policy

        self.memory = []  # Clear memory after updates

def reward_function(old_state, new_state, best_y, previous_states, repeat, start, clear):
    #--Multipliers for each part of the reward function, customise as needed
    x_a_mult = -1
    y_a_mult = 1
    y_pos_mult = 1

    #--Reward function
    reward = -0.1 #default value, punish no action
    old_x_distance = (old_state[0] - start[0]) # old distance from start horizontally
    old_y_distance = (old_state[1] - start[1]) # old distance above start  
    new_x_distance = (new_state[0] - start[0]) # new distance from start horizontally
    new_y_distance = (new_state[1] - start[1]) # new distance above start    

    x_change = new_x_distance-old_x_distance
    y_change = new_y_distance-old_y_distance
    if (new_state[1] >= best_y):
        y_change = 0
        x_change = 0

    if ((new_state[0] != old_state[0]) or (new_state[1] != old_state[1])):
        acc_reward = (x_a_mult * x_change) + (y_a_mult * y_change)
        reward = reward + acc_reward
        repeat = 0
    else:
        acc_reward = 0
        repeat = repeat + 1
        if (repeat >= 10):
            reward -= 100

    y_movement = (old_state[1] - new_state[1])
    reward = reward + (y_pos_mult * y_movement) # reward moving up, punish moving down. all levels require reaching the top of the screen

    if(new_state == [0, 0, 0, 0]): #madeline died
        reward -= 50

    elif(new_state == [100,100,100,100]): #level complete
        if (clear == 0): #this will only happen the FIRST time the agent clears the level
            clear = 1
        reward += 15000

    else: #if agent died, state reads as [0, 0, 0, 0], which would give dying a positive reward

        if (new_state[1] < best_y): #reward reaching a new highest point
            best_y = new_state[1]
            reward += 10

        if (new_state in previous_states): #punish going to an old spot
            reward -= 1

        #if (new_state[2] == 0):
            #reward -= 10
    return reward, best_y, repeat, clear

# Define the game state retrieval function
def get_game_state(p8):
    """Retrieve the current game state."""
    player = p8.game.get_player()
    #if player:
        #return [player.x, player.y, player.spd.x, player.spd.y]  # Example state
    if (type(player) == p8.game.player):
        return [player.x, player.y, player.spd.x, player.spd.y] #player in level
    if (type(player) == p8.game.player_spawn):
        return [100,100,100,100] #player completed level
    return [0, 0, 0, 0]  # Default state if player is not available, level failed

# Main Training Loop
def train_pyleste(num_episodes):
    # Initialize the PICO-8 emulator with Celeste
    global p8
    p8 = PICO8(Celeste)  # Create PICO-8 instance with Celeste

    # Define observation and action spaces
    state_dim = 4  # Example: [player_x, player_y, velocity_x, velocity_y]
    action_dim = 9  # Updated actions to include new combinations, REMOVED LEFT AND RIGHT DASH

    clear_flag = 0 #used to find when agent first clears the room

    agent = PPOAgent(state_dim, action_dim)  # Initialize PPO agent
    episode_rewards = []  # List to store episode rewards
    
    returned_episodes = [0, 1, 2, 3, 99, 100, 999, 1000, 9999,  10000, 19999, 20000, 49999, 50000, 99998, 99999]

    for episode in range(num_episodes):
        repeat_level = 0 #if agent spends too long in one state, kill it
        # Reset the environment
        p8.reset()  # Reset PICO-8 emulator
        utils.load_room(p8,0)
        utils.skip_player_spawn(p8)

        state = get_game_state(p8)  # Initialize Pyleste
        start_state = state #measure distance relative to start of level
        done = False  # Episode termination flag
        total_reward = 0  # Total reward for the episode
        action_list = []
        state_list = []
        best_y = state[1]
        action_mapping = {
            1: 0b000001,  # Press Left
            2: 0b000010,  # Press Right
            3: 0b010000,  # Press Jump
            4: 0b010001,  # Press Left + Jump
            5: 0b010010,  # Press Right + Jump
            #6: 0b100001,  # Press Left + Dash
            #7: 0b100010,  # Press Right + Dash
            6: 0b100100,  # Press Up + Dash
            7: 0b100101,  # Press Left + Up + Dash
            8: 0b100110,  # Press Right + Up + Dash
        }

        while not done:
            action, log_prob = agent.act(state)  # Select action, and execute it
            chosen_action = action_mapping.get(action,0) 
            action_list.append(chosen_action) # append action to action list for later viewing
    
            p8.set_btn_state(0b000000) #skip one frame, so agent doesnt hold down one button permanently
            p8.step()
            action_list.append(0)

            next_state = get_game_state(p8)  # Retrieve next state
            state_list.append(next_state)
            
            if (len(action_list) > 600):
                next_state = [0,0,0,0] #kill agent if game has been going on too long

            reward, new_y, repeat_level, clear_flag = reward_function(state, next_state, best_y, state_list, repeat_level, start_state, clear_flag)  # calculate reward based on reward function
            
            if (clear_flag == 1):
                print("FIRST CLEAR ON EPISODE: ", episode)
                clear_flag == 2 #makes this only happen once

            best_y = new_y #track highest point maddy has reached

            if (repeat_level >= 10):
                next_state = [0, 0, 0, 0] #kill agent if it hasn't moved for 10 frames
            done = (next_state == [0, 0, 0, 0] or next_state == [100,100,100,100]) #check for termination flag

            _, value = agent.policy(torch.FloatTensor(state).to(device))  # Get state value
            agent.store_transition((state, action, log_prob, reward, done, value.item()))  # Store transition

            state = next_state  # Update state
            total_reward += reward  # Accumulate reward
            #print(total_reward)

        agent.update_policy()  # Update policy at the end of the episode
        episode_rewards.append(total_reward)  # Append total reward
        if (episode in returned_episodes):
            #utils.load_room(p8,0)
            #utils.skip_player_spawn(p8)
            #utils.watch_inputs(p8,action_list)
            print(action_list)
            print(f"Episode {episode}: Total Reward: {total_reward}")  # Log progress
            #print(state_list)
    return episode_rewards  # Return episode rewards

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device

# Example Usage
if __name__ == "__main__":
    num_episodes = 100000  # Number of training episodes
    rewards = train_pyleste(num_episodes)  # Train PPO agent
    x_axis = [a for a in range(num_episodes)]

    plt.plot(x_axis, rewards)

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.title("Rewards per episode")

    plt.show()
    

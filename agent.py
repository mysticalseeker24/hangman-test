import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
import time
import logging
import os
import matplotlib.pyplot as plt

from model import QNetwork
from memory import ReplayMemory

class HangmanPlayer:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_actions = env.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.steps_done = 0
        self.episode_durations = []
        self.reward_in_episode = []
        self.wins = []
        self.compile()

    def compile(self):
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.training['learning_rate'])
        self.memory = ReplayMemory(self.config.rl['max_queue_length'])

    def _update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _adjust_learning_rate(self, epoch):
        lr = self.config.training['learning_rate'] * (0.1 ** (epoch // 1000))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_action_for_state(self, state, epoch=0):
        sample = random.random()
        eps_threshold = self.config.epsilon['min_epsilon'] + (self.config.epsilon['max_epsilon'] - self.config.epsilon['min_epsilon']) * \
            self.config.epsilon['decay_epsilon'] ** self.steps_done
        self.steps_done += 1
        
        # Convert state to tensor and ensure proper shape
        state_tensor = torch.tensor(state[0], device=self.device, dtype=torch.float).unsqueeze(0)
        actions_tensor = torch.tensor(state[1], device=self.device, dtype=torch.float)
        
        if sample > eps_threshold:
            with torch.no_grad():
                # Pass through Q-network to get Q-values
                q_values = self.q_network(state_tensor, actions_tensor)
                valid_actions = [i for i in range(26) if state[1][i] == 0]
                if not valid_actions:
                    return random.randint(0, 25)
                
                q_values = q_values[0, valid_actions]
                action_idx = q_values.argmax().item()
                return valid_actions[action_idx]
        else:
            guessed = set(string.ascii_lowercase[i] for i, used in enumerate(state[1]) if used)
            entropy = np.zeros(26)
            for i, letter in enumerate(string.ascii_lowercase):
                if letter not in guessed:
                    p = self.env.letter_frequencies.get(letter, 0.01)
                    entropy[i] = -p * np.log2(p + 1e-10)
            for i, letter in enumerate(string.ascii_lowercase):
                if letter in guessed:
                    entropy[i] = -float('inf')
            probs = np.exp(entropy - np.max(entropy))
            probs /= probs.sum() + 1e-10
            return np.random.choice(26, p=probs)

    def save(self, epoch):
        torch.save({
            'q_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward_in_episode': self.reward_in_episode,
            'episode_durations': self.episode_durations,
            'wins': self.wins,
            'steps_done': self.steps_done
        }, f'qlearning_hangman_epoch{epoch}.pt')
        # Save final model separately
        torch.save({
            'q_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward_in_episode': self.reward_in_episode,
            'episode_durations': self.episode_durations,
            'wins': self.wins,
            'steps_done': self.steps_done
        }, 'qlearning_hangman.pt')

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.reward_in_episode = checkpoint.get('reward_in_episode', [])
        self.episode_durations = checkpoint.get('episode_durations', [])
        self.wins = checkpoint.get('wins', [])
        self.steps_done = checkpoint.get('steps_done', 0)

    def optimize_model(self):
        if len(self.memory) < self.config.training['batch_size']:
            return 0  # Not enough samples for training

        # Sample random batch from memory
        transitions = self.memory.sample(self.config.training['batch_size'])
        batch = type(transitions[0])(*zip(*transitions))
        
        # Compute mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = [torch.tensor(s, device=self.device, dtype=torch.float) for s in batch.next_state if s is not None]
        non_final_next_actions = [torch.tensor(a, device=self.device, dtype=torch.float) for a, s in zip(batch.action, batch.next_state) if s is not None]
        
        if non_final_next_states:
            non_final_next_states_tensor = torch.stack([s[0] for s in non_final_next_states])
            non_final_next_actions_tensor = torch.stack([a for a in non_final_next_actions])

        state_batch = torch.stack([torch.tensor(s[0], device=self.device, dtype=torch.float) for s in batch.state])
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long).unsqueeze(1)
        actions_used_batch = torch.stack([torch.tensor(s[1], device=self.device, dtype=torch.float) for s in batch.state])
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.q_network(state_batch, actions_used_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.config.training['batch_size'], device=self.device)
        if non_final_next_states:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_network(non_final_next_states_tensor, non_final_next_actions_tensor).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.config.rl['gamma'] * next_state_values * (1 - done_batch))

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)  # Gradient clipping
        self.optimizer.step()

        return loss.item()

    def train(self):
        total_steps = 0
        total_epochs = self.config.training['num_epochs']
        start_time = time.time()
        losses = []
        epoch_rewards = []
        epoch_wins = []
        win_rate_history = []

        # Initialize training environment
        logging.info(f"Starting training for {total_epochs} epochs...")

        for epoch in range(total_epochs):
            epoch_loss = 0
            epoch_reward = 0
            epoch_win = 0
            num_episodes = 0
            
            # Curriculum learning - gradually increase difficulty
            difficulty = min(1.0, epoch / self.config.training['warmup_epochs']) if epoch < self.config.training['warmup_epochs'] else 1.0
            
            # Train on multiple random words
            for word_idx in range(len(self.env.wordlist)):
                if total_steps >= total_epochs * len(self.env.wordlist) * self.config.training['iterations_per_word']:
                    break
                    
                # Get a random word
                state = self.env.reset(difficulty=difficulty)
                done = False
                episode_reward = 0
                
                # Play one episode
                while not done:
                    # Select action
                    action = self._get_action_for_state(state, epoch)
                    
                    # Execute action
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Store transition in memory
                    self.memory.push(state, action, next_state if not done else None, reward, done)
                    
                    # Move to next state
                    state = next_state
                    episode_reward += reward
                    
                    # Optimize model
                    loss = self.optimize_model()
                    if loss:
                        epoch_loss += loss
                    
                    # Update target network
                    if total_steps % 1000 == 0:
                        self._update_target()
                        
                    total_steps += 1
                
                # Episode statistics
                self.episode_durations.append(self.env.max_tries - self.env.tries)
                self.reward_in_episode.append(episode_reward)
                win = info.get('win', False)
                self.wins.append(1 if win else 0)
                
                epoch_reward += episode_reward
                epoch_win += 1 if win else 0
                num_episodes += 1
                
                # Print progress
                if word_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    logging.info(f"Epoch {epoch}/{total_epochs}, Word {word_idx}, Reward: {episode_reward:.2f}, Win: {win}, Epsilon: {self.config.epsilon['min_epsilon'] + (self.config.epsilon['max_epsilon'] - self.config.epsilon['min_epsilon']) * self.config.epsilon['decay_epsilon'] ** self.steps_done:.4f}, Time: {elapsed:.1f}s")
                    
                # Exit loop if we've done enough training
                if num_episodes >= self.config.training['iterations_per_word']:
                    break
            
            # Adjust learning rate periodically
            self._adjust_learning_rate(epoch)
            
            # Log epoch statistics
            if num_episodes > 0:
                avg_loss = epoch_loss / num_episodes if num_episodes > 0 else 0
                avg_reward = epoch_reward / num_episodes if num_episodes > 0 else 0
                win_rate = epoch_win / num_episodes * 100 if num_episodes > 0 else 0
                
                losses.append(avg_loss)
                epoch_rewards.append(avg_reward)
                epoch_wins.append(epoch_win)
                win_rate_history.append(win_rate)
                
                logging.info(f"Epoch {epoch}/{total_epochs} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.2f}, Win Rate: {win_rate:.2f}%, Episodes: {num_episodes}")
                
                # Save model periodically
                if epoch % self.config.training['save_freq'] == 0 or epoch == total_epochs - 1:
                    self.save(epoch)
                    self.plot_training_history(epoch_rewards, win_rate_history, epoch)
            
            # Check if we've done enough training
            if total_steps >= total_epochs * len(self.env.wordlist) * self.config.training['iterations_per_word']:
                break
                
        # Save final model
        self.save(total_epochs)
        return epoch_rewards, win_rate_history
    
    def plot_training_history(self, rewards, win_rates, epoch=None):
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot rewards
        ax1.plot(rewards, label='Average Reward per Episode')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True)
        
        # Plot win rate
        ax2.plot(win_rates, label='Win Rate (%)', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Training Win Rate')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        filename = f"qlearning_training_history{'_epoch'+str(epoch) if epoch else ''}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

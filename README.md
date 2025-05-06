# Reinforcement Learning for Hangman

This directory contains the implementation of a Reinforcement Learning (RL) model for playing the Hangman game. The RL agent learns to select the most optimal letter at each state of the game to maximize the chances of winning.

## Approach

Our reinforcement learning approach treats the Hangman game as a Markov Decision Process (MDP) where:

- **States**: The current game state (revealed letters and remaining attempts)
- **Actions**: Choosing a letter from the alphabet (that hasn't been guessed yet)
- **Rewards**: Positive rewards for correct guesses, negative for incorrect ones, and a terminal reward for winning/losing
- **Transitions**: Deterministic based on the hidden word

## Implementation Plan

1. Define the Hangman environment with OpenAI Gym-like interface
2. Implement a Deep Q-Network (DQN) agent
3. Train the agent using experience replay
4. Evaluate the agent against the existing BiLSTM model
5. Compare different RL algorithms (DQN, A2C, PPO)

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib (for visualization)

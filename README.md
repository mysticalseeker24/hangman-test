# Reinforcement Learning for Hangman

This directory contains the implementation of a Reinforcement Learning (RL) model for playing the Hangman game. The RL agent learns to select the most optimal letter at each state of the game to maximize the chances of winning.

## Implementation Details

### Algorithm: Optimized Approximate Q-learning

- **Core Mechanism**: Uses Multi-Layer Perceptron (MLP) to approximate Q-values instead of a tabular Q-table
- **Q-Value Updates**: Based on Bellman equation: Q(s,a) ← (1−α)Q(s,a) + α(r + γ max Q(s′,a′))
- **Replay Buffer**: 100,000 capacity buffer for experience replay to stabilize learning

### State Representation

- **Encoded Pattern**: 27x27 matrix (26 letters + underscore, padded to max length 25)
- **Actions Used**: 26-dimensional binary vector (1 if letter guessed, 0 otherwise)
- **Additional Features**: Word length and number of revealed letters
- **Total dimensions**: 27×27+26+2 = 761

### Neural Network Architecture

- **Model Type**: 2-layer MLP
- **Input**: 761 dimensions (flattened state)
- **Hidden Layers**: 256 units each, ReLU activation, dropout (0.3)
- **Output**: 26 Q-values (one per letter)
- **Parameters**: ~200,000

### Training Process

- **Episodes**: ~1.25M (5,000 epochs × 10 iterations/word × ~25,000 words)
- **Curriculum Learning**: Progresses from short words (3–5 letters) to longer ones (up to 20)
- **Optimizer**: Adam with learning rate 0.001, decayed by 0.1 every 1,000 epochs
- **Target Network**: Updated every 50 epochs for stability

### Exploration Strategy

- **Epsilon-Greedy**: Starting at ε=1.0, final ε=0.01, decay rate 0.999
- **Random Action Selection**: Weighted by letter entropy based on frequency in dictionary

### Reward Structure

- **Correct Guess**: +1 + 0.5 × new_letters_revealed
- **Incorrect Guess**: −2 − 0.5 × letter_frequency
- **Repeated Guess**: −4
- **Win**: +10
- **Loss**: −5

### Performance

- **Target Win Rate**: 70-80% (potentially 85% with tuning)
- **Validation**: 200 simulated games with stratified sampling by word length
- **API Integration**: Plays 100 practice games and 1,000 recorded games

## Files

- **qlearning_hangman_solver.ipynb**: Complete implementation of the Q-learning agent for Hangman

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Gym
- Matplotlib
- Requests
- Scikit-learn

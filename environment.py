import gym
import numpy as np
import random
import string
import logging
from gym import spaces
from gym.utils import seeding
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

class HangmanEnv(gym.Env):
    def __init__(self, dictionary_path='words_250000_train.txt'):
        super(HangmanEnv, self).__init__()
        self.dictionary_path = dictionary_path
        self.wordlist = self._load_wordlist()
        
        # Calculate letter frequencies for smarter exploration
        self.letter_frequencies = self._calculate_letter_frequencies()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(26)  # a-z
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=1, shape=(25, 27), dtype=np.float32),  # Word pattern matrix
            'used_letters': spaces.MultiBinary(26)  # Which letters have been guessed
        })
        
        # Initialize environment state
        self.word = None
        self.pattern = None
        self.tries = None
        self.max_tries = 6
        self.actions_used = set()
        self.actions_correct = set()
        self.np_random = None
        self.seed()
        self.reset()
        
    def _load_wordlist(self):
        """Load word list from dictionary file"""
        with open(self.dictionary_path, 'r') as f:
            words = [w.strip().lower() for w in f.readlines() if w.strip()]
        return [w for w in words if all(c in string.ascii_lowercase for c in w)]
    
    def _calculate_letter_frequencies(self):
        """Calculate letter frequencies in the wordlist"""
        all_letters = ''.join(self.wordlist)
        counter = Counter(all_letters)
        total = sum(counter.values())
        return {char: count / total for char, count in counter.items()}
    
    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _get_pattern(self):
        """Get current word pattern based on guessed letters"""
        return ''.join([c if c in self.actions_correct else '_' for c in self.word])
    
    def _get_state(self):
        """Convert current game state to observation space format"""
        # One-hot encoding of the pattern
        encoding = np.zeros((25, 27))
        pattern = self._get_pattern().ljust(25, ' ')
        
        for i, c in enumerate(pattern[:25]):
            if c == '_':
                encoding[i][26] = 1  # Special index for unknown letters
            else:
                encoding[i][ord(c) - ord('a')] = 1
        
        # One-hot encoding of used letters
        used_letters = np.zeros(26)
        for c in self.actions_used:
            if c in string.ascii_lowercase:
                used_letters[ord(c) - ord('a')] = 1
        
        return encoding, used_letters
    
    def reset(self, difficulty=None):
        """Reset the environment for a new game"""
        self.word = random.choice(self.wordlist)
        self.tries = self.max_tries
        self.actions_used = set()
        self.actions_correct = set()
        
        # Optional curriculum learning by difficulty
        if difficulty is not None and 0 <= difficulty <= 1:
            difficulty = max(0.1, min(difficulty, 1.0))  # Clamp between 0.1 and 1.0
            num_initial = int(difficulty * len(set(self.word)))
            self._reveal_initial_letters(num_initial)
            
        self.pattern = self._get_pattern()
        state = self._get_state()
        
        # Log reset information
        logging.info(f"Reset: Word={self.word}, Guess={self.pattern}, Actions={self.actions_used}")
        
        return state
    
    def _reveal_initial_letters(self, num_correct, num_incorrect=0):
        """Reveal some letters to start with (curriculum learning)"""
        unique_letters = set(self.word)
        correct_guesses = set(random.sample(list(unique_letters), k=min(num_correct, len(unique_letters))))
        available_incorrect = [c for c in string.ascii_lowercase if c not in self.word]
        incorrect_guesses = set(random.sample(available_incorrect, k=min(num_incorrect, len(available_incorrect)))) if available_incorrect else set()
        self.actions_correct = correct_guesses
        self.actions_used = correct_guesses.union(incorrect_guesses)
    
    def step(self, action):
        """Take an action in the environment"""
        letter = chr(action + ord('a'))
        reward = 0
        done = False
        info = {'win': False, 'gameover': False}
        
        # Check if the letter has already been guessed
        if letter in self.actions_used:
            reward = -0.5  # Penalty for guessing the same letter
            done = False
        else:
            self.actions_used.add(letter)
            old_pattern = self.pattern
            
            # Check if the letter is in the word
            if letter in self.word:
                self.actions_correct.add(letter)
                new_pattern = self._get_pattern()
                revealed_count = sum(1 for a, b in zip(old_pattern, new_pattern) if a != b)
                reward = revealed_count * 1.0  # Reward proportional to letters revealed
                
                # Check if the word has been completely guessed
                if '_' not in new_pattern:
                    reward += 5.0  # Bonus for winning
                    done = True
                    info['win'] = True
                    info['gameover'] = True
            else:
                self.tries -= 1
                reward = -1.0  # Penalty for incorrect guess
                
                # Check if out of tries
                if self.tries <= 0:
                    reward -= 2.0  # Additional penalty for losing
                    done = True
                    info['gameover'] = True
        
        # Update pattern after action
        self.pattern = self._get_pattern()
        state = self._get_state()
        
        # Log step information
        logging.info(f"Step: Action={letter}, Reward={reward}, Done={done}, Guess={self.pattern}")
        
        return state, reward, done, info
    
    def render(self, mode='human'):
        """Render the current state of the environment"""
        if mode == 'human':
            print(f"Word: {self.pattern}")
            print(f"Guessed: {''.join(sorted(self.actions_used))}")
            print(f"Tries left: {self.tries}/{self.max_tries}")
        return self.pattern

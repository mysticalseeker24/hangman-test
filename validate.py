import random
import string
import numpy as np
import torch
import time
import logging
from collections import defaultdict
from environment import HangmanEnv
from agent import HangmanPlayer
from config import Config

def validate_model(player, num_games=200, verbose=False):
    """Validate the model on a set of random games"""
    env = player.env
    wins = 0
    total_guesses = 0
    results = []
    failed_games = []
    word_lengths = defaultdict(int)
    word_length_wins = defaultdict(int)
    
    # Use stratified sampling to ensure we test across different word lengths
    words_by_length = defaultdict(list)
    for word in env.wordlist:
        words_by_length[len(word)].append(word)
    
    # Select words to test with stratified sampling
    test_words = []
    for length, words in words_by_length.items():
        num_words = min(len(words), max(1, int(num_games * (len(words) / len(env.wordlist)))))
        test_words.extend(random.sample(words, num_words))
    
    # If we don't have enough words, sample more randomly
    if len(test_words) < num_games:
        remaining_words = [w for w in env.wordlist if w not in test_words]
        test_words.extend(random.sample(remaining_words, min(len(remaining_words), num_games - len(test_words))))
    
    test_words = test_words[:num_games]  # Ensure we don't have more than requested
    
    start_time = time.time()
    for i, word in enumerate(test_words):
        if verbose and (i % 10 == 0 or i + 1 == len(test_words)):
            print(f"Validating game {i+1}/{len(test_words)}")
        
        # Force the environment to use this word
        env.word = word
        env.tries = env.max_tries
        env.actions_used = set()
        env.actions_correct = set()
        env.pattern = env._get_pattern()
        state = env._get_state()
        
        word_lengths[len(word)] += 1
        done = False
        guesses = 0
        guessed_letters = []
        
        # Play the game
        while not done:
            action = player._get_action_for_state(state)
            letter = string.ascii_lowercase[action]
            guessed_letters.append(letter)
            next_state, reward, done, info = env.step(action)
            state = next_state
            guesses += 1
            
            if guesses >= env.max_tries * 2:  # Safety check for infinite loops
                done = True
                info['win'] = False
                info['gameover'] = True
        
        # Record results
        if info.get('win', False):
            wins += 1
            word_length_wins[len(word)] += 1
        else:
            failed_games.append({
                'word': word,
                'guessed_letters': guessed_letters,
                'final_pattern': env.pattern
            })
        
        total_guesses += guesses
        results.append({
            'word': word,
            'win': info.get('win', False),
            'guesses': guesses,
            'guessed_letters': guessed_letters
        })
    
    # Calculate statistics
    win_rate = (wins / len(test_words)) * 100 if test_words else 0
    avg_guesses = total_guesses / len(test_words) if test_words else 0
    elapsed_time = time.time() - start_time
    
    # Performance by word length
    length_performance = {}
    for length in word_lengths:
        length_win_rate = (word_length_wins[length] / word_lengths[length]) * 100 if word_lengths[length] > 0 else 0
        length_performance[length] = {'games': word_lengths[length], 'wins': word_length_wins[length], 'win_rate': length_win_rate}
    
    # Print results
    print(f"\nValidation Results:")
    print(f"Games: {len(test_words)}, Wins: {wins}, Win Rate: {win_rate:.2f}%")
    print(f"Average Guesses per Game: {avg_guesses:.2f}")
    print(f"Validation Time: {elapsed_time:.2f} seconds")
    print("\nPerformance by Word Length:")
    for length in sorted(length_performance.keys()):
        perf = length_performance[length]
        print(f"Length {length}: {perf['win_rate']:.2f}% win rate ({perf['wins']}/{perf['games']} games)")
    
    if failed_games:
        print("\nSample of Failed Games:")
        for r in failed_games[:5]:
            print(f"Word: {r['word']}, Guessed: {r['guessed_letters']}, Pattern: {r['final_pattern']}")
    
    return {'win_rate': win_rate, 'avg_guesses': avg_guesses, 'results': results, 'length_performance': length_performance}

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(filename='qlearning_validation.log', level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    # Load model
    config = Config()
    env = HangmanEnv(dictionary_path='words_250000_train.txt')
    player = HangmanPlayer(env, config)
    
    try:
        player.load('qlearning_hangman.pt')
        logging.info("Loaded pre-trained model for validation")
    except Exception as e:
        logging.error(f"Could not load model: {e}")
        logging.info("Please train the model first by running main.py")
        exit(1)
    
    # Run validation
    validate_model(player, num_games=200, verbose=True)

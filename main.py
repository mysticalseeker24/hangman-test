import os
import time
import logging
import torch
import matplotlib.pyplot as plt
import argparse
from config import Config
from environment import HangmanEnv
from agent import HangmanPlayer
from validate import validate_model
from api import HangmanAPI

def setup_logging():
    """Configure logging to both file and console"""
    logging.basicConfig(filename='qlearning_hangman.log', level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

def train_model(config, env, player, resume=False):
    """Train the Q-learning model"""
    # Check if model exists and should be resumed
    model_path = 'qlearning_hangman.pt'
    if resume and os.path.exists(model_path):
        player.load(model_path)
        logging.info(f"Resuming training from existing model: {model_path}")
    else:
        logging.info("Starting training from scratch")
    
    # Train the model
    start_time = time.time()
    rewards, win_rates = player.train()
    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Plot training history
    player.plot_training_history(rewards, win_rates)
    logging.info(f"Training plot saved to qlearning_training_history.png")
    
    return player

def run_api_tests(config, env, player, access_token, dictionary_path):
    """Run API tests with the trained model"""
    logging.info("Setting up API connection")
    api = HangmanAPI(access_token, 'qlearning_hangman.pt', dictionary_path, player)
    
    logging.info("Running 100 practice games")
    practice_results = api.play_games(num_games=100, practice=True, verbose=False)
    
    logging.info(f"Practice results: Win rate: {practice_results['win_rate']:.2f}%")
    
    logging.info("Running 1000 recorded games for submission")
    submission_results = api.play_games(num_games=1000, practice=False, verbose=False)
    
    logging.info(f"Submission results: Win rate: {submission_results['win_rate']:.2f}%")
    return practice_results, submission_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Q-Learning Hangman Solution')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--resume', action='store_true', help='Resume training from saved model')
    parser.add_argument('--validate', action='store_true', help='Validate model performance')
    parser.add_argument('--api-test', action='store_true', help='Run API tests')
    parser.add_argument('--dictionary', default='words_250000_train.txt', help='Path to dictionary file')
    parser.add_argument('--access-token', default='32e370374596861bcf313f8646476b', help='Trexquant API access token')
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Print system info
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check dictionary file
    if not os.path.exists(args.dictionary):
        logging.error(f"Dictionary file not found: {args.dictionary}")
        return
    
    # Initialize environment and agent
    config = Config()
    env = HangmanEnv(dictionary_path=args.dictionary)
    player = HangmanPlayer(env, config)
    
    # Run requested operations
    if args.train:
        player = train_model(config, env, player, resume=args.resume)
    elif not os.path.exists('qlearning_hangman.pt'):
        logging.warning("No trained model found and --train not specified. Model performance may be poor.")
    else:
        player.load('qlearning_hangman.pt')
        logging.info("Loaded pre-trained model")
    
    if args.validate:
        validation_results = validate_model(player, num_games=200, verbose=True)
        logging.info(f"Validation win rate: {validation_results['win_rate']:.2f}%")
    
    if args.api_test:
        practice_results, submission_results = run_api_tests(
            config, env, player, args.access_token, args.dictionary)

if __name__ == "__main__":
    main()

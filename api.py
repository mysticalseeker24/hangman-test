import requests
import string
import time
import random
import numpy as np
import torch
import logging

class HangmanAPI:
    def __init__(self, access_token, model_path, dictionary_path, player, session=None, timeout=2000):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        self.full_dictionary = self.build_dictionary(dictionary_path)
        self.current_dictionary = self.full_dictionary.copy()
        self.player = player
        self.char_to_id = {chr(97+x): x for x in range(26)}
        self.char_to_id['_'] = 26

    def determine_hangman_url(self):
        links = ['https://trexsim.com']
        data = {link: float('inf') for link in links}
        for link in links:
            try:
                start = time.time()
                requests.get(link, verify=False, timeout=2)
                data[link] = time.time() - start
            except Exception:
                continue
        link = min(data.items(), key=lambda x: x[1])[0] if any(v != float('inf') for v in data.values()) else links[0]
        return link + '/trexsim/hangman'

    def build_dictionary(self, dictionary_file):
        with open(dictionary_file, 'r') as f:
            return [word.strip().lower() for word in f.readlines() if word.strip()]

    def encode_pattern(self, pattern):
        encoding = np.zeros((25, 27))
        for i, c in enumerate(pattern[:25]):
            encoding[i][self.char_to_id[c]] = 1
        return encoding

    def guess(self, word):
        clean_word = ''.join(c for c in word.lower() if c in string.ascii_lowercase + '_')
        print(f'Current word: {word}, Guessed: {sorted(self.guessed_letters)}')
        state = (
            self.encode_pattern(clean_word),
            np.array([1 if c in self.guessed_letters else 0 for c in string.ascii_lowercase])
        )
        action = self.player._get_action_for_state(state)
        letter = string.ascii_lowercase[action]
        print(f'Predicts: {letter}')
        self.guessed_letters.append(letter)
        return letter

    def start_game(self, practice=True, verbose=True):
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary.copy()
        try:
            response = self.request('/new_game', {'practice': practice})
        except Exception as e:
            print(f'Error starting game: {e}')
            return False
        if response.get('status') == 'approved':
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print(f'Started game! ID: {game_id}, Tries: {tries_remains}, Word: {word}')
            while tries_remains > 0:
                guess_letter = self.guess(word)
                if verbose:
                    print(f'Guessing: {guess_letter}')
                try:
                    res = self.request('/guess_letter', {'request': 'guess_letter', 'game_id': game_id, 'letter': guess_letter})
                except Exception as e:
                    print(f'Request error: {e}')
                    continue
                if verbose:
                    print(f'Server response: {res}')
                word = res.get('word', word)
                tries_remains = res.get('tries_remains', 0)
                
                if '_' not in word:
                    if verbose:
                        print(f'Game won: {word}')
                    return {'result': 'win', 'word': word, 'guessed_letters': self.guessed_letters}
                elif tries_remains <= 0:
                    if verbose:
                        print(f'Game lost. Word was: {res.get("answer", "unknown")}')
                    return {'result': 'loss', 'word': word, 'answer': res.get('answer', 'unknown'), 'guessed_letters': self.guessed_letters}
        else:
            print(f'Game not approved: {response}')
            return False

    def play_games(self, num_games=100, practice=True, verbose=False):
        wins = 0
        losses = 0
        failed_games = []
        
        for game in range(1, num_games + 1):
            if game % 10 == 0 or game == 1 or game == num_games:
                print(f'Playing game {game}/{num_games}')
                
            result = self.start_game(practice=practice, verbose=verbose)
            if not result:
                continue
                
            if result['result'] == 'win':
                wins += 1
            else:
                losses += 1
                failed_games.append({
                    'game': game,
                    'last_word': result['word'],
                    'answer': result.get('answer', 'unknown'),
                    'guessed_letters': sorted(result['guessed_letters'])
                })
            
            # Add a short delay to avoid API rate limiting
            time.sleep(0.5)
        
        win_rate = wins / (wins + losses) * 100 if wins + losses > 0 else 0
        print(f'\nResults after {wins + losses} games:')
        print(f'Wins: {wins}, Losses: {losses}, Win rate: {win_rate:.2f}%')
        if failed_games:
            print('\nFailed games sample (up to 5):')
            for fg in failed_games[:5]:
                print(f"Game {fg['game']}: Guessed {fg['guessed_letters']}, Answer was: {fg['answer']}")
        return {'wins': wins, 'losses': losses, 'win_rate': win_rate, 'failed_games': failed_games}

    def request(self, path, args=None):
        url = self.hangman_url + path
        headers = {'Authorization': f'Bearer {self.access_token}'} if self.access_token else {}
        try:
            response = self.session.post(url, json=args, headers=headers, timeout=self.timeout, verify=False) if args else \
                     self.session.get(url, headers=headers, timeout=self.timeout, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f'Request failed: {e}')
            # Add retry mechanism with exponential backoff
            retries = 3
            backoff = 1
            for retry in range(retries):
                print(f'Retrying in {backoff} seconds... (Attempt {retry+1}/{retries})')
                time.sleep(backoff)
                backoff *= 2
                try:
                    response = self.session.post(url, json=args, headers=headers, timeout=self.timeout, verify=False) if args else \
                             self.session.get(url, headers=headers, timeout=self.timeout, verify=False)
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.RequestException as e2:
                    print(f'Retry {retry+1} failed: {e2}')
            raise
        except ValueError as e:
            print(f'JSON parsing error: {e}')
            return {}

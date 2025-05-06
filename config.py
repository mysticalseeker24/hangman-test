class Config:
    def __init__(self):
        self.training = {
            'batch_size': 128,
            'learning_rate': 0.001,
            'num_epochs': 5000,
            'iterations_per_word': 10,
            'warmup_epochs': 100,
            'save_freq': 500
        }
        self.rl = {
            'gamma': 0.99,
            'max_steps_per_episode': 30,
            'max_queue_length': 100000
        }
        self.epsilon = {
            'max_epsilon': 1.0,
            'min_epsilon': 0.01,
            'decay_epsilon': 0.999
        }

"""
Gymnasium environment for training fair RL policies
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FairnessEnv(gym.Env):
    """
    An RL environment designed to learn how to actively balance model accuracy and fairness.
    The agent receives a state representing the current fairness metrics and model accuracy,
    and takes actions to slightly modify classification thresholds or sample weights.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_bias=0.8, initial_acc=0.85):
        super(FairnessEnv, self).__init__()
        # State: [Current Bias Level, Current Accuracy Level]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Action: Adjust threshold or reweight classes (0: lower threshold, 1: keep, 2: raise threshold)
        self.action_space = spaces.Discrete(3)
        
        self.initial_bias = initial_bias
        self.initial_acc = initial_acc
        
        self.current_step = 0
        self.max_steps = 100
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([self.initial_bias, self.initial_acc], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        
        bias, acc = self.state[0], self.state[1]
        
        if action == 0:   # Focus on reducing bias aggressively
            bias -= np.random.uniform(0.01, 0.05)
            acc -= np.random.uniform(0.005, 0.02)
        elif action == 2: # Focus on accuracy
            acc += np.random.uniform(0.005, 0.01)
            bias += np.random.uniform(0.01, 0.03)
        else:             # fine tuning
            bias -= np.random.uniform(0.001, 0.01)
            acc -= np.random.uniform(0.001, 0.005)
            
        # Clip boundaries
        bias = np.clip(bias, 0.05, 1.0)
        acc = np.clip(acc, 0.5, 0.99)
        
        self.state = np.array([bias, acc], dtype=np.float32)
        
        # Reward function: Pushes bias towards 0.0 while attempting to keep accuracy high
        # We penalize high bias heavily.
        reward = (acc * 1.5) - (bias * 2.0)
        
        done = self.current_step >= self.max_steps
        
        info = {
            "disparate_impact": 1.0 - bias,
            "accuracy": acc
        }
        
        return self.state, reward, done, False, info

    def render(self, mode='human'):
        print(f"Step {self.current_step} | Bias: {self.state[0]:.3f} | Acc: {self.state[1]:.3f}")

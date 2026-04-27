"""
PPO Agent to dynamically solve fairness constraints.
"""
from stable_baselines3 import PPO
from rl.env import FairnessEnv
import os

def train_ppo_agent(episodes: int = 100, save_path: str = "models/ppo_fairness_model"):
    """
    Initializes and trains a PPO model against the FairnessEnv.
    """
    if not os.path.exists("models"):
        os.makedirs("models")
        
    print(f"Initializing PPO Sandbox Gym for {episodes} steps...")
    env = FairnessEnv()
    
    # Init PPO
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
    
    # Train
    max_timesteps = episodes * env.max_steps
    model.learn(total_timesteps=max_timesteps)
    
    # Save optimized model
    model.save(save_path)
    print(f"Training Complete. Policy rules injected. Saved at {save_path}.zip")
    
    return model

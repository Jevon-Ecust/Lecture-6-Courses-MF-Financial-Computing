# source: Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. ICAIF 2020
import pandas as pd
import numpy as np
import time
import gym
import tensorflow as tf
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.deepq.policies import DQNPolicy
from utilities import *
from EnvMultipleStock_train import StockEnvTrain
from EnvMultipleStock_trade import StockEnvTrade


def train_DQN(env_train, learning_rate, model_name, timesteps=25000):
    """DQN model"""
    model = DQN('MlpPolicy', env_train, learning_rate=learning_rate,
                exploration_fraction=0.1, param_noise=False, verbose=0, tensorboard_log='log')
    model.learn(total_timesteps=timesteps)
    model.save(f"trained_models/{model_name}")
    return model


def run_model() -> None:
    filename = 'dow_30_2009_2020.csv'
    data = preprocess_data(filename)
    print(data.head())
    print(data.size)
    df = data
    train = data_split(df, start=20100101, end=20151231)
    env_train = StockEnvTrain(train)
    learning_rate = 0.0001
    for its in range(1):
        model_DQN = train_DQN(env_train, learning_rate, model_name="DQN_30k_128_dow_{}".format(its), timesteps=300000)


if __name__ == "__main__":
    run_model()

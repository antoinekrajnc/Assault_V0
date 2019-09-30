import gym 
from gym.wrappers.monitor import Monitor
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


env = gym.make("Assault-v0")
env = Monitor(env, "videos", force=True)


# test 
img = env.reset() # Donne des images en (250, 160, 3)

### Test du modèle 
from model import Model
model = Model(num_actions=env.action_space.n)
action, value = model.action_value(obs = img)


# Creation of the Agent

class Agent():
    def __init__(self, model):

        self.params = {"value": 1/7,
                       "entropy": 0.0001,
                       "gamma": 0.75}

        self.model = model 

        self.model.compile(tf.keras.optimizers.Adam(lr=0.0001),
                           loss = [self._logits_loss, self._value_loss])

    ### Bof compris l'idée d'advantages
    #### Returns is the sum of cumulative rewards 
    #### Advantages is the difference between values of the state and the cumulated rewards of taking an action
    def _return_advantages(self, rewards, dones, values, next_value):

        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params["gamma"] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]

        advantages = returns - values

        return returns, advantages

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        
        while not done:

            action, _ = self.model.action_value(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()

        return ep_reward

    def _value_loss(self, returns, value):
        return self.params["value"] * tf.keras.losses.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)

        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        actions = tf.cast(actions, tf.int32)

        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        
        return policy_loss - self.params["entropy"]*entropy_loss

    def train(self, env, batch_size=32, updates=1000):

        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + env.observation_space.shape)

        ep_rews = [0.0]
        next_obs = env.reset()

        for update in range(updates):
            for step in range(batch_size):

                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs)
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rews[-1] += rewards[step]

                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs)
            returns, advs = self._return_advantages(rewards, dones, values, next_value)

            acts_and_advs = np.concatenate([actions[: , None], advs[:, None]], axis=-1)
            
            losses = self.model.fit(observations, 
                                    [acts_and_advs, returns], 
                                    batch_size=batch_size,
                                    verbose=0,
                                    )

            print(update, ep_rews[-1])
        return ep_rews

agent = Agent(model)

# Entrainement
rewards_history = agent.train(env, 32, 100)

# Test
agent.test(env)
env.close()
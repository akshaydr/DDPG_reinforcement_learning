# DDPG_reinforcement_learning
Deep Deterministic Policy Gradient is an algorithm for deep reinforcement learning with continuous space. It is used for environments with continuous action space. It is an extension of Deep-Q-Network (DQN) where DQN is from continuous state space and discrete action space. The algorithm concurrently learns a Q-function and a policy. It uses off-policy data and Bellman equation to learn the Q-function, and uses the Q-function to learn the policy. 

In a continuous action environment finding the maximum over actions very difficult. In other words, it is difficult to solve the optimization problem. However, policy gradient methods and actor-critic methods mitigate the problem by looking for a local optimum by Bellman methods.

## History
DDPG is derived from two algorithms Deep-Q-Network (DQN) and Deterministic Policy Gradients (DPG). DPG is an efficient gradient computation for deterministic policies. DQN has few features such as reply buffer and target-q-network which are used to define DDPG algorithm. Combining these two algorithms will give the DDPG algorithm. 

## Architecture
The DDPG has two networks which are actor and critic as shown in the figure. The critic takes sets of inputs which are states and actions and returns Q-value of that state and of that action. Whereas, the actor takes set of states as input and returns a set of actions. All the updates are performed based on stochastic gradient decent. 

## Experimental Results
![Mountain Car Continuous](mountainCar.gif)

### Applied Hyper Parameters:
    ACTOR LEARNING RATE: 0.0001
    CRITIC LEARNING RATE: 0.001
    CRITIC UPDATE DISCOUNT FACTOR: 0.99
    SOFT TARGET UPDATE PARAMETER: 0.01
    MINI-BATCH SIZE: 64

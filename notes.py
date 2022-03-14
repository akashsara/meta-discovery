"""
EpsGreedyQPolicy: Standard epsilon greedy policy
LinearAnnealedPolicy: Linearly anneals the given attribute
This effectively just reduces the value of epsilon over time.
It starts from max and slowly goes to min. 
At testing time epsilon is set to 0.
"""

"""
DQNAgent:
Model, policy, memory and nb_actions are self-explanatory.
Warm Up: 
We use a very low LR for a fixed number of steps.
Then, we increase it to the actual LR. This is done to both
reduce overall variance and also combat overfitting.
Especially in a case where the first few batches might be 
highly correlated data which might skew the model.
Gamma = Discount Factor

target_model_update:
if x >= 1, update model (target_model = model) every x steps
Else do a "soft update" to shift the target_model to the model.
So over x steps we'll arrive at the model.
Refer: https://github.com/keras-rl/keras-rl/issues/55

delta_clip = Delta for clipping gradients.
Refer: https://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/

enable_double_dqn = Have a secondary target network
According to the paper "Deep Reinforcement Learning with Double Q-learning"
(van Hasselt et al., 2015), in Double DQN, the online network predicts the 
actions while the target network is used to estimate the Q value.
This model receives no updates, but instead clones the original's weights.

train_interval = After how many steps do we perform a training update?
This is different from the target model update. This updates our neural network. Target Model Update updates our secondary target network. 

memory_interval = After how many steps do we add something to the memory.
"""

"""
SequentialMemory: 
limit = How many entries to hold. 
        Once we hit this number, we replace the oldest ones.
window_length = How many observations need to be concatenated to form a state. 
                For our purposes this should always be 1.
"""
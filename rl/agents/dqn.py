import torch
import torch.nn as nn
import sys
from tqdm import tqdm
import numpy as np
import os

sys.path.append("./")
from rl.memory import Transition
import graphics


class DQNAgent:
    def __init__(
        self,
        policy,
        memory,
        model,
        optimizer,
        loss,
        model_kwargs={},
        optimizer_kwargs={},
        loss_kwargs={},
        batch_size=32,
        gamma=0.95,
        use_soft_update=True,
        tau=1e-3,
        train_interval=1,
        log_interval=100,
        warmup_steps=1000,
        load_dict_path=None,
    ):
        # Setup hyperparameters
        self.iterations = 0
        self.gamma = gamma
        self.use_soft_update = use_soft_update
        self.tau = tau
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.log_interval = log_interval
        self.warmup_steps = warmup_steps

        self.policy = policy
        self.memory = memory

        # Setup device
        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")

        # Setup policy & target networks
        self.policy_network = model(**model_kwargs)
        self.target_network = model(**model_kwargs)

        # target network uses the policy network's weights
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Only policy network is trained
        self.policy_network.train()
        self.target_network.eval()

        # Setup optimizer
        self.optimizer = optimizer(self.policy_network.parameters(), **optimizer_kwargs)

        # Setup loss function
        self.loss_function = loss(**loss_kwargs)

        # Move models to GPU if possible
        self.policy_network.to(self.device)
        self.target_network.to(self.device)

        # Setup some variables to track things
        self.losses = []
        self.rewards = []
        self.battle_lengths = []

        if load_dict_path:
            print("Loading Model")
            load_dict = torch.load(load_dict_path, map_location=self.device)
            self.policy_network.load_state_dict(load_dict["model_state_dict"])
            self.target_network.load_state_dict(load_dict["model_state_dict"])
            self.optimizer.load_state_dict(load_dict["optimizer_state_dict"])
            print("Load successful.")

    def fit(self, environment, num_training_steps):
        state = environment.reset()
        self.policy_network.train()
        self.target_network.eval()
        all_rewards = []
        all_losses = []
        all_battle_lengths = []
        loss = None
        current_battle_length = 0
        for i in tqdm(range(num_training_steps)):
            # Make action mask
            action_mask = environment.get_action_mask()
            # Get q_values
            with torch.no_grad():
                q_values = self.policy_network(state)
            # Use policy
            action = int(self.policy.select_action(q_values, action_mask))
            # Play move
            next_state, reward, done, info = environment.step(action)

            if done:
                next_state = None

            # Save transition in memory
            self.memory.push(state, action, next_state, reward, action_mask)
            # Reset environment if we're done
            if done:
                all_battle_lengths.append(current_battle_length)
                current_battle_length = 0
                state = environment.reset()
            # Else just use the next state as the current state
            else:
                state = next_state
                current_battle_length += 1

            # Train model on one batch of data
            if (self.iterations > self.warmup_steps) and (
                self.iterations % self.train_interval == 0
            ):
                loss = self.train()

            # Use a soft update
            if self.use_soft_update:
                self.soft_update()
            # Update target network every tau steps
            elif self.iterations % self.tau == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

            # Store metrics
            all_rewards.append(reward)
            if loss:
                all_losses.append(loss)
                loss = None

            # Housekeeping
            self.iterations += 1

            # Log output to console
            if self.iterations % self.log_interval == 0:
                tqdm.write(
                    f"[{i + 1}/{num_training_steps}] Iteration: {self.iterations}\tAverage Reward: {np.mean(all_rewards)}\tAverage Loss: {np.mean(all_losses)}"
                )
        self.rewards.extend(all_rewards)
        self.losses.extend(all_losses)
        self.battle_lengths.extend(all_battle_lengths)

    def train(self):
        # Only train if we have enough samples for a batch of data
        if len(self.memory) < self.batch_size:
            return

        # Create batch from memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Separate out data into separate batches
        # (batch_size, state_size)
        state_batch = torch.stack(batch.state).float().to(self.device)
        # (batch_size, 1)
        action_batch = torch.tensor(batch.action).unsqueeze(dim=1).to(self.device)
        # (batch_size, )
        reward_batch = torch.tensor(batch.reward).to(self.device)

        # Final states would be None, so we mask those out as the value is 0 there
        non_final_mask = torch.tensor(
            tuple(map(lambda state: state is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        # Handle edge cases where all the samples are final states
        if not any(non_final_mask):
            return

        # Compute the next state transitions for non-final states
        non_final_next_states = []
        non_final_action_masks = []
        for mask, state in zip(batch.action_mask, batch.next_state):
            if state is not None:
                non_final_next_states.append(state)
                non_final_action_masks.append(mask)

        # (non_final_batch_size, state_size)
        non_final_next_states = (
            torch.stack(non_final_next_states).float().to(self.device)
        )
        # (non_final_batch_size, n_actions)
        non_final_action_masks = (
            torch.stack(non_final_action_masks).float().to(self.device)
        )

        # Compute Q(s_t, a)
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Which is just max_a Q(s'_t+1, a)
        # We compute this using the target_network for stability.
        # V(s) = 0 for all final states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            (self.target_network(non_final_next_states) + non_final_action_masks)
            .max(1)[0]
            .detach()
        )

        # Compute the expected Q values: (max_a Q(s_t+1, a) * gamma) + reward
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.loss_function(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().cpu()

    def soft_update(self):
        for policy_params, target_params in zip(
            self.policy_network.parameters(), self.target_network.parameters()
        ):
            target_params.data.copy_(
                self.tau * policy_params.data + (1.0 - self.tau) * target_params.data
            )

    def test(self, environment, num_episodes):
        self.policy_network.eval()
        for episode in tqdm(range(num_episodes)):
            done = False
            state = environment.reset()
            while not done:
                action_mask = environment.get_action_mask()
                # Get q_values
                with torch.no_grad():
                    q_values = self.policy_network(state)
                # Use policy
                action = int(self.policy.greedy_action(q_values, action_mask))
                # Play move
                state, reward, done, info = environment.step(action)

    def save(self, output_path):
        x = np.array(self.rewards)
        average_rewards = x.cumsum() / (np.arange(x.size) + 1)
        x = np.array(self.losses)
        average_losses = x.cumsum() / (np.arange(x.size) + 1)
        x = np.array(self.battle_lengths)
        average_battle_length = x.cumsum() / (np.arange(x.size) + 1)
        graphics.plot_and_save_loss(
            average_rewards, "steps", "reward", os.path.join(output_path, "reward.jpg")
        )
        graphics.plot_and_save_loss(
            average_losses, "steps", "loss", os.path.join(output_path, "loss.jpg")
        )
        graphics.plot_and_save_loss(
            average_battle_length,
            "episodes",
            "battle_length",
            os.path.join(output_path, "battle_length.jpg"),
        )
        torch.save(
            {
                "model_state_dict": self.policy_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(output_path, "model.pt"),
        )
        torch.save(
            {
                "loss": self.losses,
                "reward": self.rewards,
                "battle_length": self.battle_lengths,
            },
            os.path.join(output_path, "statistics.pt"),
        )

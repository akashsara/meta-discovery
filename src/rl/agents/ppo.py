# https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L257
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm

sys.path.append("./")
import graphics
from rl.memory import PPOMemory


class PPOAgent:
    def __init__(
        self,
        model,
        optimizer,
        steps_per_rollout,
        state_size,
        n_actions,
        memory_size,
        model_kwargs={},
        optimizer_kwargs={},
        batch_size=32,
        num_training_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        value_clip_param=0.2,
        c1=0.5,
        c2=0.001,
        normalize_advantages=False,
        use_action_mask=True,
        load_dict_path=None,
    ):
        # Setup training hyperparameters
        self.iterations = 0
        self.batch_size = batch_size
        self.num_training_epochs = num_training_epochs
        self.use_action_mask = use_action_mask

        # Setup model hyperparameters
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_clip_param = value_clip_param
        self.c1 = c1
        self.c2 = c2
        self.normalize_advantages = normalize_advantages

        # Setup device
        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")

        # Setup policy & target networks
        self.model = model(**model_kwargs)

        # Fix the number of steps per rollout
        self.steps_per_rollout = steps_per_rollout

        # Ensure that batch size is a factor of memory size
        if self.steps_per_rollout % self.batch_size != 0:
            raise ValueError("steps_per_rollout must be divisible by batch_size.")

        # Setup memory
        self.memory = PPOMemory(
            batch_size=self.batch_size,
            gamma=self.gamma,
            gae_lambda=gae_lambda,
            memory_size=memory_size,
            state_size=state_size,
            n_actions=n_actions,
        )

        # We use this to track episode states if we cut off mid-episode
        self.last_episode_start = False

        # Variables to track episode statistics
        self.current_episode_length = 0
        self.current_episode_returns = 0

        # Setup optimizer
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)

        # Move models to GPU if possible
        self.model.to(self.device)

        # Setup some variables to track things
        self.rewards = []
        self.episode_returns = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy = []
        self.total_losses = []
        self.approx_kl_divs = []
        self.time_taken_per_rollout = []

        # Print model
        print(self.model)

        if load_dict_path:
            print("Loading Model")
            load_dict = torch.load(load_dict_path, map_location=self.device)
            self.model.load_state_dict(load_dict["model_state_dict"])
            self.optimizer.load_state_dict(load_dict["optimizer_state_dict"])
            self.iterations = load_dict["iterations"]
            print("Load successful.")

    def fit(self, environment, total_steps, do_training=True):
        self.model.eval()
        total_iterations = self.iterations + total_steps
        if total_steps % self.steps_per_rollout != 0:
            raise ValueError("Total steps must be divisible by steps_per_rollout.")
        num_rollouts = total_steps // self.steps_per_rollout
        # Fresh start - Ignore previously running battles
        state = environment.reset()
        self.last_episode_start = False
        self.current_episode_length = 0
        self.current_episode_returns = 0
        for rollout in range(num_rollouts):
            start = time.time()
            # Clear memory so that we have fresh rollouts
            self.memory.clear()
            # Gather fresh data
            current_step = rollout * self.steps_per_rollout
            print(
                f"Gathering: Current: [{current_step}/{total_steps}]\tTotal: [{self.iterations}/{total_iterations}]"
            )
            episode_idx = len(self.episode_returns)
            reward_idx = len(self.rewards)
            state = self.collect_rollouts(environment, state)
            # Log information to console
            print(
                f"Mean Episode Reward: {np.mean(self.episode_returns[episode_idx:]):.4f}\tMean Reward: {np.mean(self.rewards[reward_idx:]):.4f}\tMean Episode Length: {np.mean(self.episode_lengths[episode_idx:]):.2f}"
            )
            # Run PPO Training
            if do_training:
                print(f"PPO Training: [{rollout+1}/{num_rollouts}]")
                loss_idx = len(self.total_losses)
                self.train()
                print(f"Mean Loss: {np.mean(self.total_losses[loss_idx:]):.4f}")
            time_taken = time.time() - start
            self.time_taken_per_rollout.append(time_taken)
            print(f"Rollout [{rollout+1}/{num_rollouts}] Completed in {time_taken:.4f}s")

    def collect_rollouts(self, environment, state):
        for step in range(self.steps_per_rollout):
            # Don't use transition in memory
            # TODO: Remove after bugfix
            if environment.skip_current_step():
                print("SKIP STEP")
                action = 0
                next_state, reward, done, _ = environment.step(action)
            else:
                with torch.no_grad():
                    state = state.to(self.device)
                    action_mask = None
                    if self.use_action_mask:
                        action_mask = environment.action_masks()
                    # Get policy & value
                    policy, value = self.model(state)
                    # Get policy distribution
                    distribution = self.get_distribution(policy, action_mask)
                    # Sample action
                    action = distribution.sample().detach()
                    # Get log probabilities
                    log_probs = distribution.log_prob(action)
                    action = int(action)
                next_state, reward, done, _ = environment.step(action)
                self.memory.push(
                    state.cpu(),
                    action,
                    reward,
                    self.last_episode_start,
                    value.cpu(),
                    log_probs.cpu(),
                    action_mask,
                )
            # Handle state transitions
            if done:
                state = environment.reset()
            else:
                state = next_state
            # Track stuff for future
            self.last_episode_start = done
            self.iterations += 1
            self.current_episode_length += 1
            self.current_episode_returns += reward
            self.rewards.append(reward)

            # Add episodic metric trackers
            if done:
                self.episode_lengths.append(self.current_episode_length)
                self.episode_returns.append(self.current_episode_returns)
                self.current_episode_length = 0
                self.current_episode_returns = 0

        with torch.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            state = state.to(self.device)
            _, value = self.model(state)
        # Compute GAE
        self.memory.compute_returns_and_advantage(last_value=value, done=done)
        return state

    def get_distribution(self, policy, action_mask=None):
        """Stochastic Action Selection"""
        # Apply action mask if it exists
        if action_mask is not None:
            policy = policy + action_mask.to(self.device)
        # Create distribution
        distribution = Categorical(probs=policy.softmax(dim=-1))
        return distribution

    def get_action(self, policy, action_mask=None):
        """Deterministic Action Selection"""
        # Apply action mask if it exists
        if action_mask is not None:
            policy = policy + action_mask.to(self.device)
        # Sample action
        action = policy.argmax(dim=-1).detach().cpu().numpy()
        return action

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.num_training_epochs)):
            # Do a complete pass through the memory
            for batch in self.memory.sample():
                # Retrieve transitions
                states = torch.tensor(batch["states"]).to(self.device)
                old_values = torch.tensor(batch["values"]).to(self.device)
                actions = torch.tensor(batch["actions"]).squeeze().to(self.device)
                action_masks = torch.tensor(batch["action_masks"]).to(self.device)
                old_log_probs = (
                    torch.tensor(batch["log_probs"]).squeeze().to(self.device)
                )
                returns = torch.tensor(batch["returns"]).to(self.device)
                advantages = torch.tensor(batch["advantages"]).squeeze().to(self.device)

                # Normalize advantages
                if self.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Get current policy distribution & values
                policy, values = self.model(states)
                if self.use_action_mask:
                    distribution = self.get_distribution(policy, action_masks)
                else:
                    distribution = self.get_distribution(policy)
                new_log_probs = distribution.log_prob(actions)

                # Calculate entropy
                entropy = distribution.entropy().mean()
                entropy = -(self.c2 * entropy)

                # Calculate Actor Loss
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate Clipped Critic Loss
                clipped_values = old_values + torch.clamp(
                    values - old_values, -self.value_clip_param, self.value_clip_param
                )
                clipped_values = nn.functional.mse_loss(returns, clipped_values)
                critic_loss = self.c2 * clipped_values

                # Gradient Ascent: Actor Loss - c1*Critic Loss + c2*Entropy
                # Gradient Descent = -Gradient Ascent
                loss = actor_loss + critic_loss + entropy

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                ## Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                # Calculate approximate form of reverse KL-Divergence
                # for early stopping. Adapted from:
                # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L257
                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).cpu()

                # Store loss values for future plotting
                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())
                self.entropy.append(entropy.item())
                self.total_losses.append(loss.item())
                self.approx_kl_divs.append(approx_kl_div.item())

    def test(self, environment, num_episodes):
        self.model.eval()
        all_rewards = []
        episode_rewards = []
        for _ in tqdm(range(num_episodes)):
            done = False
            state = environment.reset()
            episode_reward = 0
            while not done:
                action_mask = None
                if self.use_action_mask:
                    action_mask = environment.action_masks()
                # Get q_values
                with torch.no_grad():
                    policy, value = self.model(state.to(self.device))
                # Use policy
                action = int(self.get_action(policy, action_mask))
                # Play move
                state, reward, done, _ = environment.step(action)
                all_rewards.append(reward)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        return np.mean(all_rewards), np.mean(episode_rewards)

    def plot_and_save_metrics(
        self, output_path, is_cumulative=False, reset_trackers=False, create_plots=True
    ):
        if is_cumulative:
            suffix = "final"
        else:
            suffix = str(self.iterations)
        # Plot trackers
        if create_plots:
            x = np.array(self.rewards)
            average_rewards = x.cumsum() / (np.arange(x.size) + 1)
            x = np.array(self.episode_lengths)
            average_episode_length = x.cumsum() / (np.arange(x.size) + 1)
            x = np.array(self.episode_returns)
            average_episode_return = x.cumsum() / (np.arange(x.size) + 1)
            x = np.array(self.actor_losses)
            actor_loss = x.cumsum() / (np.arange(x.size) + 1)
            x = np.array(self.critic_losses)
            critic_loss = x.cumsum() / (np.arange(x.size) + 1)
            x = np.array(self.entropy)
            entropy = x.cumsum() / (np.arange(x.size) + 1)
            x = np.array(self.total_losses)
            total_loss = x.cumsum() / (np.arange(x.size) + 1)
            x = np.array(self.approx_kl_divs)
            approx_kl_divs = x.cumsum() / (np.arange(x.size) + 1)
            graphics.plot_and_save_loss(
                average_rewards,
                "steps",
                "reward",
                os.path.join(output_path, f"reward_{suffix}.jpg"),
            )
            graphics.plot_and_save_loss(
                average_episode_length,
                "episodes",
                "episode_length",
                os.path.join(output_path, f"episode_length_{suffix}.jpg"),
            )
            graphics.plot_and_save_loss(
                average_episode_return,
                "episodes",
                "episode_return",
                os.path.join(output_path, f"episode_return_{suffix}.jpg"),
            )
            graphics.plot_and_save_loss(
                actor_loss,
                "steps",
                "actor loss",
                os.path.join(output_path, f"actor_loss_{suffix}.jpg"),
            )
            graphics.plot_and_save_loss(
                critic_loss,
                "steps",
                "critic loss",
                os.path.join(output_path, f"critic_loss_{suffix}.jpg"),
            )
            graphics.plot_and_save_loss(
                entropy,
                "steps",
                "entropy",
                os.path.join(output_path, f"entropy_{suffix}.jpg"),
            )
            graphics.plot_and_save_loss(
                total_loss,
                "steps",
                "total loss",
                os.path.join(output_path, f"total_loss_{suffix}.jpg"),
            )
            graphics.plot_and_save_loss(
                approx_kl_divs,
                "steps",
                "approximate KL-divergence",
                os.path.join(output_path, f"approx_kl_divs_{suffix}.jpg"),
            )
            graphics.plot_and_save_loss(
                self.time_taken_per_rollout,
                "rollout",
                "time_taken",
                os.path.join(output_path, f"time_taken_per_rollout_{suffix}.jpg"),
            )
        # Save trackers
        torch.save(
            {
                "reward": torch.tensor(self.rewards),
                "episode_lengths": torch.tensor(self.episode_lengths),
                "episode_returns": torch.tensor(self.episode_returns),
                "actor_loss": torch.tensor(self.actor_losses),
                "critic_loss": torch.tensor(self.critic_losses),
                "entropy": torch.tensor(self.entropy),
                "total_loss": torch.tensor(self.total_losses),
                "approx_kl_divs": torch.tensor(self.approx_kl_divs),
                "time_taken_per_rollout": torch.tensor(self.time_taken_per_rollout),
            },
            os.path.join(output_path, f"statistics_{suffix}.pt"),
        )
        if reset_trackers:
            self.rewards = []
            self.episode_lengths = []
            self.episode_returns = []
            self.actor_losses = []
            self.critic_losses = []
            self.entropy = []
            self.total_losses = []
            self.approx_kl_divs = []
            self.time_taken_per_rollout = []

    def save(self, output_path, reset_trackers=False, create_plots=True):
        self.plot_and_save_metrics(
            output_path, reset_trackers=reset_trackers, create_plots=create_plots
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iterations": self.iterations
            },
            os.path.join(output_path, f"model_{self.iterations}.pt"),
        )

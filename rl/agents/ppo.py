# https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import sys
from tqdm import tqdm
import numpy as np
import os

sys.path.append("./")
from rl.memory import PPOTransition
import graphics


class PPOAgent:
    def __init__(
        self,
        model,
        optimizer,
        memory,
        model_kwargs={},
        optimizer_kwargs={},
        batch_size=32,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        c1=0.5,
        c2=0.001,
        log_interval=100,
        load_dict_path=None,
    ):
        # Setup training hyperparameters
        self.iterations = 0
        self.batch_size = batch_size
        self.log_interval = log_interval

        # Setup model hyperparameters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_param = clip_param
        self.c1 = c1
        self.c2 = c2

        self.memory = memory

        # Setup device
        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")

        # Setup policy & target networks
        self.model = model(**model_kwargs)

        # Setup optimizer
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)

        # Move models to GPU if possible
        self.model.to(self.device)

        # Setup some variables to track things
        self.rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy = []
        self.approx_kl_divs = []
        self.total_losses = []

        # Print model
        print(self.model)

        if load_dict_path:
            print("Loading Model")
            load_dict = torch.load(load_dict_path, map_location=self.device)
            self.model.load_state_dict(load_dict["model_state_dict"])
            self.optimizer.load_state_dict(load_dict["optimizer_state_dict"])
            print("Load successful.")

    def fit(self, environment, steps_per_epoch, num_epochs, do_training=True):
        total_iterations = self.iterations + (steps_per_epoch * num_epochs)
        for epoch in range(num_epochs):
            self.model.eval()
            start_iterations = self.iterations
            # Play steps_per_rollout steps
            # Note that we may go slightly above this as we prioritize
            # completing an episode so we can calculate the returns
            print(f"PPO Gathering: [{epoch+1}/{num_epochs}]")
            while (self.iterations - start_iterations) < steps_per_epoch:
                state = environment.reset()
                done = False
                entropy = 0
                episode_length = 0
                episode_states = []
                episode_rewards = []
                episode_actions = []
                episode_action_masks = []
                episode_log_probs = []
                episode_values = []
                episode_done_mask = []
                episode_return = 0
                # Play one full episode
                while not done:
                    if environment.skip_current_step():
                        print("SKIP STEP")
                        action = 0
                        next_state, reward, done, _ = environment.step(action)
                    else:
                        # Make action mask
                        action_mask = environment.get_action_mask().to(self.device)
                        # Get policy & value
                        policy, value = self.model(state.to(self.device))
                        # Get policy distribution
                        distribution = self.get_distribution(policy, action_mask)
                        # Sample action
                        action = distribution.sample().detach()
                        # Get log probabilities
                        log_probs = distribution.log_prob(action)
                        # Get entropy
                        step_entropy = distribution.entropy().mean()
                        # Play Move
                        next_state, reward, done, _ = environment.step(int(action))
                        # Store variables needed for learning
                        episode_return += reward
                        entropy += step_entropy
                        episode_states.append(state)
                        episode_rewards.append(reward)
                        episode_actions.append(action)
                        episode_action_masks.append(action_mask)
                        episode_log_probs.append(log_probs)
                        episode_values.append(value)
                        episode_done_mask.append(1 - done)
                        # Store for later logging/graphs
                        self.rewards.append(reward)
                    # Transition to next state
                    state = next_state
                    # Housekeeping
                    episode_length += 1
                    self.iterations += 1
                    # Log output to console
                    if self.iterations % self.log_interval == 0:
                        print(
                            f"[{self.iterations}/{total_iterations}] Average Reward: {np.mean(self.rewards):.4f}\tAverage Episode Length: {np.mean(self.episode_lengths):.2f}\tAverage Loss: {np.mean(self.total_losses):.4f}"
                        )
                self.episode_lengths.append(episode_length)
                # Compute GAE at the end of the episode
                next_state = next_state.to(self.device)
                _, next_value = self.model(next_state)
                episode_returns = self.compute_advantage(
                    next_value, episode_rewards, episode_done_mask, episode_values
                )
                # Organize data for storage
                returns = torch.stack(episode_returns).detach()
                log_probs = torch.stack(episode_log_probs).detach()
                values = torch.stack(episode_values).detach()
                states = torch.stack(episode_states)
                actions = torch.stack(episode_actions)
                action_masks = torch.stack(episode_action_masks).detach()
                advantages = returns - values
                # Store episode information in memory
                self.memory.push(
                    states, actions, action_masks, log_probs, returns, advantages
                )
            # Run PPO Training
            if do_training:
                print(f"PPO Training: [{epoch+1}/{num_epochs}]")
                self.train()
                # Clear memory
                self.memory.clear()

    def get_distribution(self, policy, action_mask=None):
        """Stochastic Action Selection"""
        # Apply action mask if it exists
        if action_mask:
            policy = policy + action_mask
        # Create distribution
        distribution = Categorical(probs=policy.softmax(dim=-1))
        return distribution

    def get_action(self, policy, action_mask=None):
        """Deterministic Action Selection"""
        # Apply action mask if it exists
        if action_mask:
            policy = policy + action_mask
        # Sample action
        action = policy.argmax(dim=-1).detach().cpu().numpy()
        return action

    def compute_advantage(self, next_values, rewards, masks, values):
        """Uses Generalized Advantage Estimation"""
        values = values + [next_values]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + (self.gamma * values[step + 1] * masks[step])
                - values[step]
            )
            gae = delta + self.gamma * self.lambda_ * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def train(self):
        self.model.train()
        self.memory.generate_batches()
        num_batches = self.memory.get_num_batches()
        for batch in tqdm(self.memory.sample(), total=num_batches):
            # Retrieve transitions
            batch = PPOTransition(*zip(*batch))
            states = torch.stack(batch.state).to(self.device)
            actions = torch.tensor(batch.action).to(self.device)
            action_masks = torch.stack(batch.action_mask).to(self.device)
            old_log_probs = torch.stack(batch.log_prob).to(self.device)
            returns = torch.tensor(batch.return_).unsqueeze(dim=1).to(self.device)
            advantages = torch.tensor(batch.advantage).unsqueeze(dim=1).to(self.device)

            # advantages = returns - old_values
            # So: old_values = returns - advantages
            old_values = returns - advantages

            # Get current policy distribution & values
            policy, values = self.model(states)
            distribution = self.get_distribution(policy, action_masks)

            # Calculate entropy
            entropy = distribution.entropy().mean()
            entropy = self.c2 * entropy

            # Calculate Actor Loss
            # Get log probabilities of the previously selected actions
            # On the current version of the model
            # I.E. pi_new(a_t|s_t)
            new_log_probs = distribution.log_prob(actions)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * advantages
            )
            actor_loss = torch.min(surr1, surr2).mean()

            # Calculate Clipped Critic Loss
            clipped_values = old_values + (values - old_values).clip(
                -self.clip_param, self.clip_param
            )
            values = nn.functional.mse_loss(returns, values)
            clipped_values = nn.functional.mse_loss(returns, clipped_values)
            critic_loss = self.c2 * torch.max(values, clipped_values)

            # Gradient Ascent: Actor Loss - c1*Critic Loss + c2*Entropy
            # Gradient Descent = -Gradient Ascent
            loss = -1 * (actor_loss - critic_loss + entropy)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            ## Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5)
            self.optimizer.step()

            # Calculate approximate form of reverse KL-Divergence 
            # for early stopping
            # Adapted from: 
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L257
            with torch.zero_grad():
                log_ratio = new_log_probs - old_log_probs
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

            # Store loss values for future plotting
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
            self.entropy.append(entropy.item())
            self.total_losses.append(loss.item())
            self.approx_kl_divs.append(approx_kl_div)

    def test(self, environment, num_episodes):
        self.model.eval()
        all_rewards = []
        episode_rewards = []
        for _ in tqdm(range(num_episodes)):
            done = False
            state = environment.reset()
            episode_reward = 0
            while not done:
                action_mask = environment.get_action_mask().to(self.device)
                # Get q_values
                with torch.no_grad():
                    value, policy = self.model(state.to(self.device))
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
        # Save trackers
        torch.save(
            {
                "reward": torch.tensor(self.rewards),
                "episode_lengths": torch.tensor(self.episode_lengths),
                "actor_loss": torch.tensor(self.actor_losses),
                "critic_loss": torch.tensor(self.critic_losses),
                "approx_kl_divs": torch.tensor(self.approx_kl_divs),
                "entropy": torch.tensor(self.entropy),
                "total_loss": torch.tensor(self.total_losses),
            },
            os.path.join(output_path, f"statistics_{suffix}.pt"),
        )
        if reset_trackers:
            self.rewards = []
            self.episode_lengths = []
            self.actor_losses = []
            self.critic_losses = []
            self.entropy = []
            self.approx_kl_divs = []
            self.total_losses = []

    def save(self, output_path, reset_trackers=False, create_plots=True):
        self.plot_and_save_metrics(
            output_path, reset_trackers=reset_trackers, create_plots=create_plots
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(output_path, f"model_{self.iterations}.pt"),
        )

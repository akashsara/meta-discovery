import os
import sys

import torch

sys.path.append("./")
from common_utils import *


def poke_env_model_evaluation(player, model, num_episodes):
    average_reward, episodic_average_reward = model.test(
        player, num_episodes=num_episodes
    )
    print(
        f"Evaluation: {player.n_won_battles} victories out of {num_episodes} episodes. Average Reward: {average_reward:.4f}. Average Episode Reward: {episodic_average_reward:.4f}"
    )
    return player.n_won_battles, average_reward, episodic_average_reward


def poke_env_validate_model(
    env_player,
    model,
    num_episodes,
    random_player,
    max_player,
    smax_player,
    key,
    results,
):
    print("Results against random player:")
    env_player.reset_env(restart=True, opponent=random_player)
    (
        random_wins,
        random_average_reward,
        random_episodic_average_reward,
    ) = poke_env_model_evaluation(env_player, model, num_episodes)

    print("\nResults against max player:")
    env_player.reset_env(restart=True, opponent=max_player)
    (
        max_wins,
        max_average_reward,
        max_episodic_average_reward,
    ) = poke_env_model_evaluation(env_player, model, num_episodes)

    print("\nResults against smart max player:")
    env_player.reset_env(restart=True, opponent=smax_player)
    (
        smax_wins,
        smax_average_reward,
        smax_episodic_average_reward,
    ) = poke_env_model_evaluation(env_player, model, num_episodes)

    results[key] = {
        "num_episodes": num_episodes,
        "vs_random": random_wins,
        "vs_max": max_wins,
        "vs_smart_max": smax_wins,
    }
    env_player.close(purge=False)
    return results


def gym_env_validate_model(results, env, model, episodes, key):
    average_rewards, average_episode_rewards = model.test(env, episodes)
    results[f"{key}"] = {
        "n_episodes": episodes,
        "average_rewards": average_rewards,
        "average_episode_rewards": average_episode_rewards,
    }
    print(f"{key} rewards: {average_rewards}, {average_episode_rewards}")
    return results


def load_trackers_to_ppo_model(output_dir, ppo):
    # Load back all the trackers to draw the final plots
    all_rewards = []
    all_episode_lengths = []
    all_episode_returns = []
    all_actor_losses = []
    all_critic_losses = []
    all_entropy = []
    all_losses = []
    all_approx_kl_divs = []
    time_taken_per_rollout = []
    # Sort files by iteration for proper graphing
    files_to_read = sorted(
        [
            int(file.split(".pt")[0].split("_")[1])
            for file in os.listdir(output_dir)
            if "statistics_" in file
        ]
    )
    for file in files_to_read:
        x = torch.load(
            os.path.join(output_dir, f"statistics_{file}.pt"), map_location=ppo.device
        )
        all_rewards.append(x.get("reward", []))
        all_episode_lengths.append(x.get("episode_lengths", []))
        all_episode_returns.append(x.get("episode_returns", []))
        all_actor_losses.append(x.get("actor_loss", []))
        all_critic_losses.append(x.get("critic_loss", []))
        all_entropy.append(x.get("entropy", []))
        all_losses.append(x.get("total_loss", []))
        all_approx_kl_divs.append(x.get("approx_kl_divs", []))
        time_taken_per_rollout.append(x.get("time_taken_per_rollout", []))
    all_rewards = torch.cat(all_rewards).flatten().cpu().numpy()
    all_episode_lengths = torch.cat(all_episode_lengths).flatten().cpu().numpy()
    all_episode_returns = torch.cat(all_episode_returns).flatten().cpu().numpy()
    all_actor_losses = torch.cat(all_actor_losses).flatten().cpu().numpy()
    all_critic_losses = torch.cat(all_critic_losses).flatten().cpu().numpy()
    all_entropy = torch.cat(all_entropy).flatten().cpu().numpy()
    all_losses = torch.cat(all_losses).flatten().cpu().numpy()
    all_approx_kl_divs = torch.cat(all_approx_kl_divs).flatten().cpu().numpy()
    time_taken_per_rollout = torch.cat(time_taken_per_rollout).flatten().cpu().numpy()
    ppo.rewards = all_rewards
    ppo.episode_lengths = all_episode_lengths
    ppo.episode_returns = all_episode_returns
    ppo.actor_losses = all_actor_losses
    ppo.critic_losses = all_critic_losses
    ppo.entropy = all_entropy
    ppo.total_losses = all_losses
    ppo.approx_kl_divs = all_approx_kl_divs
    ppo.time_taken_per_rollout = time_taken_per_rollout

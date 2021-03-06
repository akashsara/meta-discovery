import os
import torch
from collections import namedtuple


def generate_server_configuration(port):
    ServerConfiguration = namedtuple(
        "ServerConfiguration", ["server_url", "authentication_url"]
    )
    LocalhostServerConfiguration = ServerConfiguration(
        f"localhost:{port}", "https://play.pokemonshowdown.com/action.php?"
    )
    return LocalhostServerConfiguration


def poke_env_validate_model(
    env_player,
    evaluation_func,
    model,
    num_episodes,
    random_player,
    max_player,
    smax_player,
    key,
    results,
):
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=evaluation_func,
        opponent=random_player,
        env_algorithm_kwargs={
            "model": model,
            "num_episodes": num_episodes,
        },
    )
    random_wins = env_player.n_won_battles

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=evaluation_func,
        opponent=max_player,
        env_algorithm_kwargs={
            "model": model,
            "num_episodes": num_episodes,
        },
    )
    max_wins = env_player.n_won_battles

    print("\nResults against smart max player:")
    env_player.play_against(
        env_algorithm=evaluation_func,
        opponent=smax_player,
        env_algorithm_kwargs={
            "model": model,
            "num_episodes": num_episodes,
        },
    )
    smax_wins = env_player.n_won_battles
    results[key] = {
        "num_episodes": num_episodes,
        "vs_random": random_wins,
        "vs_max": max_wins,
        "vs_smart_max": smax_wins,
    }
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
        all_rewards.append(x["reward"])
        all_episode_lengths.append(x["episode_lengths"])
        all_episode_returns.append(x["episode_returns"])
        all_actor_losses.append(x["actor_loss"])
        all_critic_losses.append(x["critic_loss"])
        all_entropy.append(x["entropy"])
        all_losses.append(x["total_loss"])
        all_approx_kl_divs.append(x["approx_kl_divs"])
    all_rewards = torch.cat(all_rewards).flatten().cpu().numpy()
    all_episode_lengths = torch.cat(all_episode_lengths).flatten().cpu().numpy()
    all_episode_returns = torch.cat(all_episode_returns).flatten().cpu().numpy()
    all_actor_losses = torch.cat(all_actor_losses).flatten().cpu().numpy()
    all_critic_losses = torch.cat(all_critic_losses).flatten().cpu().numpy()
    all_entropy = torch.cat(all_entropy).flatten().cpu().numpy()
    all_losses = torch.cat(all_losses).flatten().cpu().numpy()
    all_approx_kl_divs = torch.cat(all_approx_kl_divs).flatten().cpu().numpy()
    ppo.rewards = all_rewards
    ppo.episode_lengths = all_episode_lengths
    ppo.episode_returns = all_episode_returns
    ppo.actor_losses = all_actor_losses
    ppo.critic_losses = all_critic_losses
    ppo.entropy = all_entropy
    ppo.total_losses = all_losses
    ppo.approx_kl_divs = all_approx_kl_divs

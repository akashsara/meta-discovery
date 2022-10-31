# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py

import asyncio
import json
import os
import sys
from threading import Thread

sys.path.append("./")
import training_utils as utils


async def start_battles(player1, player2, num_challenges):
    await asyncio.gather(
        player1.agent.accept_challenges(player2.username, num_challenges),
        player2.agent.send_challenges(player1.username, num_challenges),
    )


def training_function(player, opponent, model, model_kwargs):
    # Fit (train) model as necessary.
    model.fit(player, **model_kwargs)
    player.done_training = True
    # Play out the remaining battles so both fit() functions complete
    # We use 99 to give the agent an invalid option so it's forced
    # to take a random legal action
    while not opponent.done_training:
        _, _, done, _ = player.step(99)
        if done and not opponent.done_training:
            _ = player.reset()
            done = False

    # Forfeit any ongoing battles
    while player.current_battle and not player.current_battle.finished:
        _ = player.step(-1)


def run_normalplay(
    rl_model,
    env_player,
    test_player,
    random_agent,
    max_damage_agent,
    smart_max_damage_agent,
    nb_training_steps,
    validate_every,
    nb_validation_episodes,
    nb_evaluation_episodes,
    evaluation_results,
    output_dir,
):
    # Pre-Evaluation
    if nb_validation_episodes > 0:
        evaluation_results = utils.poke_env_validate_model(
            test_player,
            rl_model,
            nb_validation_episodes,
            random_agent,
            max_damage_agent,
            smart_max_damage_agent,
            f"initial",
            evaluation_results,
        )
        # Save evaluation results
        with open(os.path.join(output_dir, "results.json"), "w") as fp:
            json.dump(evaluation_results, fp)

    # Training
    num_epochs = max(nb_training_steps // validate_every, 1)
    for i in range(num_epochs):
        # Train Model
        env_player.start_challenging()
        rl_model.fit(env_player, validate_every, do_training=True)
        # Shutdown training agent
        env_player.close(purge=False)

        # Evaluate Model
        if nb_validation_episodes > 0 and i + 1 != num_epochs:
            # Save model
            rl_model.save(output_dir, reset_trackers=True, create_plots=False)
            # Validation
            evaluation_results = utils.poke_env_validate_model(
                test_player,
                rl_model,
                nb_validation_episodes,
                random_agent,
                max_damage_agent,
                smart_max_damage_agent,
                f"validation_{i+1}",
                evaluation_results,
            )
            # Save evaluation results
            with open(os.path.join(output_dir, "results.json"), "w") as fp:
                json.dump(evaluation_results, fp)

    # Save final model
    rl_model.save(output_dir, reset_trackers=True, create_plots=False)

    # Final Evaluation
    if nb_evaluation_episodes > 0:
        evaluation_results = utils.poke_env_validate_model(
            test_player,
            rl_model,
            nb_evaluation_episodes,
            random_agent,
            max_damage_agent,
            smart_max_damage_agent,
            f"final",
            evaluation_results,
        )
        # Save evaluation results
        with open(os.path.join(output_dir, "results.json"), "w") as fp:
            json.dump(evaluation_results, fp)


def run_selfplay(
    rl_model,
    player1,
    player2,
    test_player,
    random_agent,
    max_damage_agent,
    smart_max_damage_agent,
    nb_training_steps,
    validate_every,
    nb_validation_episodes,
    nb_evaluation_episodes,
    evaluation_results,
    output_dir,
):
    # Pre-Evaluation
    if nb_validation_episodes > 0:
        evaluation_results = utils.poke_env_validate_model(
            test_player,
            rl_model,
            nb_validation_episodes,
            random_agent,
            max_damage_agent,
            smart_max_damage_agent,
            f"{rl_model.iterations}_initial",
            evaluation_results,
        )
        # Save evaluation results
        with open(os.path.join(output_dir, "results.json"), "w") as fp:
            json.dump(evaluation_results, fp)

    # Training
    num_epochs = max(nb_training_steps // validate_every, 1)
    for i in range(num_epochs):
        # Setup arguments to pass to the training function
        p1_env_kwargs = {
            "total_steps": validate_every,
            "do_training": True,
        }
        p2_env_kwargs = {
            "total_steps": validate_every,
            "do_training": False,
        }

        # Self-Play bits
        player1.done_training = False
        player2.done_training = False
        # 1. Get event loop
        loop = asyncio.get_event_loop()
        # Make Two Threads; one per player and run model.fit()
        t1 = Thread(
            target=lambda: training_function(player1, player2, rl_model, p1_env_kwargs)
        )
        t1.start()

        t2 = Thread(
            target=lambda: training_function(player2, player1, rl_model, p2_env_kwargs)
        )
        t2.start()
        # On the network side, keep sending & accepting battles
        while not player1.done_training or not player2.done_training:
            loop.run_until_complete(start_battles(player1, player2, 1))
        # Wait for thread completion
        t1.join()
        t2.join()

        player1.close(purge=False)
        player2.close(purge=False)

        # Evaluate Model
        if nb_validation_episodes > 0 and i + 1 != num_epochs:
            # Save model
            rl_model.save(output_dir, reset_trackers=True, create_plots=False)
            # Validation
            evaluation_results = utils.poke_env_validate_model(
                test_player,
                rl_model,
                nb_validation_episodes,
                random_agent,
                max_damage_agent,
                smart_max_damage_agent,
                f"{rl_model.iterations}_validation",
                evaluation_results,
            )
            # Save evaluation results
            with open(os.path.join(output_dir, "results.json"), "w") as fp:
                json.dump(evaluation_results, fp)

    # Save final model
    rl_model.save(output_dir, reset_trackers=True, create_plots=False)

    # Final Evaluation
    if nb_evaluation_episodes > 0:
        evaluation_results = utils.poke_env_validate_model(
            test_player,
            rl_model,
            nb_evaluation_episodes,
            random_agent,
            max_damage_agent,
            smart_max_damage_agent,
            f"{rl_model.iterations}_final",
            evaluation_results,
        )
        # Save evaluation results
        with open(os.path.join(output_dir, "results.json"), "w") as fp:
            json.dump(evaluation_results, fp)

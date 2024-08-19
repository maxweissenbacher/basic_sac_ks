# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import time
import hydra
import torch
import torch.cuda
import tqdm
import numpy as np
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
import wandb
from torchrl.record.loggers import generate_exp_name
from sac.agents_sac import (
    log_metrics_offline,
    log_metrics_wandb,
    make_collector,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
)
from utils.rng import env_seed
from env.ks_env_utils import make_parallel_ks_env, make_ks_eval_env


# @hydra.main(version_base="1.2", config_path="", config_name="config_sac")
def main(cfg: "DictConfig"):  # noqa: F821
    # Create logger
    exp_name = generate_exp_name("SAC", cfg.env.exp_name)
    if cfg.logger.project_name is None:
        raise ValueError("WandB project name must be specified in config.")
    wandb.init(
        mode=str(cfg.logger.mode),
        project=str(cfg.logger.project_name),
        entity=str(cfg.logger.team_name),
        name=exp_name,
        config=dict(cfg),
    )

    print('Starting experiment ' + exp_name)

    torch.manual_seed(env_seed(cfg))
    np.random.seed(env_seed(cfg))

    # Create environments
    train_env = make_parallel_ks_env(cfg)
    eval_env = make_ks_eval_env(cfg)

    # Create agent
    model, exploration_policy, device = make_sac_agent(cfg, train_env, eval_env)

    # Create SAC loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    # Create off-policy collector
    collector = make_collector(cfg, train_env, exploration_policy)

    # Create replay buffer
    replay_buffer = make_replay_buffer(cfg)

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_sac_optimizer(cfg, loss_module)

    # Main loop
    start_time = time.time()
    collected_frames = 0
    # pbar = tqdm.tqdm(total=cfg.collector.total_frames // cfg.env.frame_skip)
    num_console_updates = 1000

    init_random_frames = cfg.collector.init_random_frames // cfg.env.frame_skip
    num_updates = int(
        cfg.env.num_envs
        * cfg.collector.frames_per_batch
        * cfg.optim.utd_ratio
    )
    prb = cfg.replay_buffer.prb

    sampling_start = time.time()
    train_start_time = sampling_start
    for i, tensordict in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        # Update weights of the inference policy
        collector.update_policy_weights_()

        tensordict = tensordict.reshape(-1)
        current_frames = tensordict.numel()
        # Add to replay buffer
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # Console update
        # pbar.update(tensordict.numel())
        if collected_frames % (cfg.collector.total_frames // (cfg.env.frame_skip * num_console_updates) + 1) == 0:
            console_output = f'Frame {collected_frames}/{cfg.collector.total_frames // cfg.env.frame_skip}'
            time_passed = time.time() - train_start_time
            console_output += f' | {time_passed/60:.0f} min' if time_passed/60 > 1 else f' | <1 min'
            print(console_output)

        # Optimization steps
        training_start = time.time()
        if collected_frames >= init_random_frames:
            losses = TensorDict({}, batch_size=[num_updates])
            for j in range(num_updates):
                # Sample from replay buffer
                sampled_tensordict = replay_buffer.sample().clone()
                sampled_tensordict = sampled_tensordict.to(device)

                # Compute loss
                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                # Update actor
                optimizer_actor.zero_grad()
                actor_loss.backward(retain_graph=False)
                optimizer_actor.step()

                # Update critic
                optimizer_critic.zero_grad()
                q_loss.backward()
                optimizer_critic.step()

                # Update alpha
                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                optimizer_alpha.step()

                losses[j] = loss_td.select(
                    "loss_actor", "loss_qvalue", "loss_alpha"
                ).detach()

                # Update qnet_target params
                target_net_updater.step()

                # Update priority
                if prb:
                    replay_buffer.update_tensordict_priority(sampled_tensordict)

        training_time = time.time() - training_start
        episode_end = (
            tensordict.get(("next", "done"), None)
            if tensordict.get(("next", "done"), False).any()
            else tensordict.get(("next", "terminated"), False)
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Logging
        if len(episode_rewards) > 0:
            # we never call this for some reason...
            episode_length = tensordict["next", "step_count"][episode_end]
            log_info["train/reward"] = (episode_rewards/episode_length).mean().item()
            log_info["train/last_reward"] = tensordict["next", "reward"][episode_end].mean().item()
            log_info["train/episode_length"] = cfg.env.frame_skip * episode_length.sum().item() / len(episode_length)
        if collected_frames >= init_random_frames:
            log_info["train/q_loss"] = losses.get("loss_qvalue").mean().item()
            log_info["train/actor_loss"] = losses.get("loss_actor").mean().item()
            log_info["train/alpha_loss"] = losses.get("loss_alpha").mean().item()
            log_info["train/alpha"] = loss_td["alpha"].item()
            log_info["train/entropy"] = loss_td["entropy"].item()
            log_info["train/sampling_time"] = sampling_time
            log_info["train/training_time"] = training_time

        # Evaluation
        if i % cfg.logger.eval_iter == 0:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_start = time.time()
                eval_rollout = eval_env.rollout(
                    cfg.logger.test_episode_length,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_time = time.time() - eval_start
                # Compute total reward (norm of solution + norm of actuation)
                eval_reward = eval_rollout["next", "reward"].mean(-2).mean().item()
                last_reward = eval_rollout["next", "reward"][..., -1, :].mean().item()
                # Compute u component of reward
                eval_reward_u = - torch.linalg.norm(eval_rollout["next", "u"], dim=-1).mean(-1).mean().item()
                last_reward_u = - torch.linalg.norm(eval_rollout["next", "u"][..., -1, :], dim=-1).mean().item()
                # Compute mean and std of actuation
                mean_actuation = torch.linalg.norm(eval_rollout["action"], dim=-1).mean(-1).mean().item()
                std_actuation = torch.linalg.norm(eval_rollout["action"], dim=-1).std(-1).mean().item()
                # Compute length of rollout
                terminated = eval_rollout["terminated"].nonzero()
                if terminated.nelement() > 0:
                    rollout_episode_length = terminated[0][0].item()
                else:
                    rollout_episode_length = cfg.logger.test_episode_length
                # Compute CAE accuracy during evaluation rollout
                if cfg.env.path_to_cae_model:
                    cae_output = eval_rollout["cae_output"].detach().cpu().numpy()
                    u = eval_rollout["u"].detach().cpu().numpy()
                    cae_rel_error = np.linalg.norm(cae_output - u) / np.linalg.norm(u)
                    cae_abs_error = np.linalg.norm(cae_output - u)

            log_info.update(
                {
                    "eval/reward": eval_reward,
                    "eval/last_reward": last_reward,
                    "eval/mean_actuation": mean_actuation,
                    "eval/std_actuation": std_actuation,
                    "eval/time": eval_time,
                    "eval/episode_length": rollout_episode_length,
                }
            )
            if cfg.env.path_to_cae_model:
                log_info.update(
                    {
                        "eval/cae_relative_L2_error": cae_rel_error,
                        "eval/cae_absolute_L2_error": cae_abs_error,
                    }
                )

        wandb.log(data=log_info, step=collected_frames)
        sampling_start = time.time()

    # Save replay buffer
    if cfg.logger.save_replay_buffer:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
        replay_buffer.dumps(output_dir + 'replay_buffer_SAC')
        print(f"Saved replay buffer. (Saved at {output_dir + 'replay_buffer_SAC'}).")

    collector.shutdown()
    wandb.finish()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()

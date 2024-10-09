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
from env.ks_env_utils import make_parallel_ks_env, make_ks_eval_env, make_parallel_ks_eval_env


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
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'

    print('Starting experiment ' + exp_name)

    torch.manual_seed(env_seed(cfg))
    np.random.seed(env_seed(cfg))

    # Create environments
    train_env = make_parallel_ks_env(cfg)
    eval_env = make_parallel_ks_eval_env(cfg)

    # Create agent
    model, device = make_sac_agent(cfg, train_env, eval_env)
    if cfg.logger.load_model:
        filepath = output_dir + '../../../' + cfg.logger.model_dir + 'model.pkl'
        with open(filepath, 'rb') as file:
            model_params = torch.load(file)
        model.load_state_dict(model_params)
        print(f"Model loaded from {filepath}")
    else:
        raise ValueError(f"Must load a trained model in order to run eval.")


    log_info = {}
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
        eval_rewards = eval_rollout["next", "reward"].squeeze()  # 20 x 150
        # eval_rewards = eval_rollout["next", "reward"].mean(-2)  # 20 x 1
        # last_rewards = eval_rollout["next", "reward"][..., -1, :]  # 20 x 1
        # mean_eval_reward = eval_rewards.mean().item()  # across all eval envs
        # std_eval_reward = eval_rewards.std().item()  # across all eval envs
        # mean_last_reward = last_rewards.mean().item()
        # std_last_reward = last_rewards.std().item()
        # Compute mean and std of actuation
        actuations = eval_rollout["action"].mean(dim=-1)  # 20 x 150
        abs_actuations = eval_rollout["action"].abs().mean(dim=-1)  # 20 x 150
        std_actuations = eval_rollout["action"].std(dim=-1)  # 20 x 150
        # mean_actuations = torch.linalg.norm(eval_rollout["action"], dim=-1).mean(-1)  # 20 x 1
        # std_actuations = torch.linalg.norm(eval_rollout["action"], dim=-1).std(-1)
        # mean_mean_actuation = mean_actuations.mean().item()
        # mean_std_actuation = std_actuations.mean().item()
        # Compute length of rollout
        terminated = eval_rollout["terminated"].nonzero()
        if terminated.nelement() > 0:
            rollout_episode_length = terminated[0][0].item()
        else:
            rollout_episode_length = cfg.logger.test_episode_length
    
    for step in range(cfg.logger.test_episode_length):
        for env_id in range(cfg.logger.num_eval_envs):
            log_info.update({f"eval/reward_{env_id}": eval_rewards[env_id, step].item()})
            log_info.update({f"eval/mean_actuation_{env_id}": actuations[env_id, step].item()})
            log_info.update({f"eval/mean_abs_actuation_{env_id}": abs_actuations[env_id, step].item()})
            log_info.update({f"eval/std_actuation_{env_id}": std_actuations[env_id, step].item()})
    
        wandb.log(data=log_info, step=step)
    
    wandb.finish()

    print(eval_rewards.shape)
    print(actuations.shape) 

    print(f"Evaluation took {eval_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
    
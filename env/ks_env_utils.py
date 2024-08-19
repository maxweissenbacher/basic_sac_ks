import torch.nn
import torch.optim
import numpy as np
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    InitTracker,
    FiniteTensorDictCheck,
    TransformedEnv,
    VecNorm,
    ParallelEnv,
    Compose,
    ObservationNorm,
    EnvCreator,
)
from env.KS_environment import KSenv
from utils.rng import env_seed


# ====================================================================
# Environment utils
# --------------------------------------------------------------------
def add_env_transforms(env, cfg):
    transform_list = [
        InitTracker(),
        RewardSum(),
        StepCounter(cfg.collector.max_episode_length // cfg.env.frame_skip),
        FiniteTensorDictCheck(),
    ]

    transforms = Compose(*transform_list)
    return TransformedEnv(env, transforms)


def make_ks_env(cfg):
    # Set environment hyperparameters
    device = cfg.collector.device
    actuator_locs = torch.tensor(
        np.linspace(
            start=0.0,
            stop=2 * torch.pi,
            num=cfg.env.num_actuators,
            endpoint=False
        ),
        device=device
    )
    sensor_locs = torch.tensor(
        np.linspace(start=0.0,
                    stop=2 * torch.pi,
                    num=cfg.env.num_sensors,
                    endpoint=False
                    ),
        device=device
    )
    env_params = {
        "nu": float(cfg.env.nu),
        "actuator_locs": actuator_locs,
        "sensor_locs": sensor_locs,
        "burn_in": int(cfg.env.burnin),
        "frame_skip": int(cfg.env.frame_skip),
        "soft_action": bool(cfg.env.soft_action),
        "actuator_loss_weight": 0.0,
        "actuator_scale": float(cfg.env.actuator_scale),
        "device": cfg.collector.device,
        "target": cfg.env.target,
        "N": cfg.env.N,
        "dt": cfg.env.dt,
    }

    # Create environments
    train_env = add_env_transforms(KSenv(**env_params), cfg)
    train_env.set_seed(env_seed(cfg))
    return train_env


def make_parallel_ks_env(cfg):
    make_env_fn = EnvCreator(lambda: make_ks_env(cfg))
    env = ParallelEnv(cfg.env.num_envs, make_env_fn)
    return env


def make_ks_eval_env(cfg):
    device = cfg.collector.device
    actuator_locs = torch.tensor(
        np.linspace(
            start=0.0,
            stop=2 * torch.pi,
            num=cfg.env.num_actuators,
            endpoint=False
        ),
        device=device
    )
    sensor_locs = torch.tensor(
        np.linspace(start=0.0,
                    stop=2 * torch.pi,
                    num=cfg.env.num_sensors,
                    endpoint=False
                    ),
        device=device
    )
    env_params = {
        "nu": float(cfg.env.nu),
        "actuator_locs": actuator_locs,
        "sensor_locs": sensor_locs,
        "burn_in": int(cfg.env.burnin),
        "frame_skip": int(cfg.env.frame_skip),
        "soft_action": bool(cfg.env.soft_action),
        "actuator_loss_weight": 0.0,
        "actuator_scale": float(cfg.env.actuator_scale),
        "device": cfg.collector.device,
        "target": cfg.env.target,
        "N": cfg.env.N,
        "dt": cfg.env.dt,
    }
    test_env = add_env_transforms(KSenv(**env_params), cfg)
    test_env.eval()
    return test_env

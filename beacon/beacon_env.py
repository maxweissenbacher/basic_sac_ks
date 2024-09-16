from torchrl.envs import GymWrapper
import importlib


def make_beacon_env(name):
    module = importlib.import_module(f"{name}.{name}")
    obj = getattr(module, name)
    gym_env = obj()
    return GymWrapper(gym_env)


if __name__ =='__main__':
    name = 'lorenz'
    torchrl_env = make_beacon_env(name)
    print(torchrl_env)
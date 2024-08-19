from hydra import compose, initialize
from sac.training import main
import hydra


@hydra.main(version_base="1.2", config_path="sac/", config_name="config_sac")
def train(cfg: "DictConfig"):
    main(cfg)


if __name__ == '__main__':
    train()


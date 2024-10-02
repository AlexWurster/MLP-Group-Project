import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import huggingface_hub
import hydra
from omegaconf import OmegaConf

from src.configs.configs import TrainingConfigs, register_base_configs
from src.trainer import Trainer


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(configs: TrainingConfigs) -> None:
    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    # if missing_keys:
    #     raise RuntimeError(f"Missing in config:\n{missing_keys}")

    trainer = Trainer(configs)
    _ = trainer.train()
    # _ = trainer.evaluate()

if __name__ == "__main__":
    register_base_configs()
    main()

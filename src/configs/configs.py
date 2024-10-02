from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING



# @dataclass
# class DataConfigs:
#     local_data_dir: str = MISSING
#     # train_data_filename: str = MISSING
#     # valid_data_filename: str = MISSING
#     # test_data_filename: str = MISSING
#     data_loader_configs: dict = MISSING
#     save_args: dict = MISSING


@dataclass
class ModelConfigs:
    model_name: str = MISSING
    configs: dict = MISSING
    model_args: dict = MISSING 


@dataclass
class TrainerConfigs:
    name: str = MISSING
    # configs: dict = MISSING
    # wandb_args: dict = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING
    num_epochs: int = MISSING
    loss_fn: str = MISSING
    lr: float = MISSING
    save_checkpoint_path: str = MISSING
    save_preds_path: str = MISSING
    target_dir: str = MISSING
    data_dir: str = MISSING
    patience: str = MISSING
    delta: float = MISSING
    
    wandb_args: dict = MISSING


@dataclass
class TrainingConfigs:
    # data: DataConfigs = MISSING
    trainer: TrainerConfigs = MISSING
    model: ModelConfigs = MISSING
    debug: bool = False
    random_seed: int = 42


def register_base_configs() -> None:
    configs_store = ConfigStore.instance()
    configs_store.store(name="base_config", node=TrainingConfigs)
    # configs_store.store(group="data", name="base_data_config", node=DataConfigs)
    configs_store.store(group="model", name="base_model_config", node=ModelConfigs)
    configs_store.store(
        group="trainer", name="base_trainer_config", node=TrainerConfigs
    )

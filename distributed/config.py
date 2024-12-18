from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelConfig:
    batch_size: int
    num_epochs: int
    lr: float
    local_rank: int = -1  # Rank of the local process (gpu) on a single machine
    global_rank: int = -1 # Rank of the global process across the cluster


def get_default_config() -> ModelConfig:
    return ModelConfig(
        batch_size=16,
        num_epochs=60,
        lr=10**-4,
    )

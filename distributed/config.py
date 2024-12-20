import glob
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    arch: str
    batch_size: int
    num_epochs: int
    lr: float
    checkpoint_dir: str
    model_name: str
    seed: int
    evaluate: bool
    dataset: str
    resume: str
    exp_group: str
    local_rank: int = -1  # Rank of the local process (gpu) on a single machine
    global_rank: int = -1  # Rank of the global process across the cluster


def get_default_config() -> ModelConfig:
    return ModelConfig(
        arch="alexnet",
        batch_size=16,
        num_epochs=60,
        lr=10**-4,
        checkpoint_dir="checkpoints",
        model_name="{arch}_{dataset}_{epoch:03d}.pth",
        seed=42,
        evaluate=False,
        dataset="dummy",
        resume="latest",
        exp_group="exp1_alexnet"
    )


def get_latest_checkpoint(config: ModelConfig) -> str:
    """Convenience wrapper to get the latest checkpoint."""
    return get_checkpoint(config, -1)


def get_checkpoint(config: ModelConfig, epoch: int) -> str:
    """Get checkpoint path for specific epoch or latest if epoch=-1"""
    checkpoint_dir = Path(config.checkpoint_dir)

    if epoch == -1:
        model_files = Path(config.checkpoint_dir).glob(f"*.pt")
        model_files = sorted(model_files, key=lambda x: int(x.stem.split("_")[-1]))
        if len(model_files) == 0:
            return None
        return str(model_files[-1])
    else:
        # Return specific epoch checkpoint
        filename = config.model_name.format(epoch=epoch, arch=config.arch, dataset=config.dataset)
        checkpoint_path = checkpoint_dir / filename
        return str(checkpoint_path)

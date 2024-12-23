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
    num_classes: int
    save_checkpoint: bool
    local_rank: int = -1  # Rank of the local process (gpu) on a single machine
    global_rank: int = -1  # Rank of the global process across the cluster


def get_default_config() -> ModelConfig:
    config = ModelConfig(
        arch="alexnet",
        batch_size=16,
        num_epochs=60,
        lr=10**-4,
        checkpoint_dir="checkpoints/{arch}-{dataset}/",
        model_name="{arch}_{dataset}_{epoch:03d}.pth",
        seed=42,
        evaluate=False,
        dataset="dummy",
        resume="latest",
        exp_group="exp1_alexnet",
        num_classes=10,
        save_checkpoint=False,
    )
    config.checkpoint_dir = config.checkpoint_dir.format(arch=config.arch, dataset=config.dataset)
    return config


def get_latest_checkpoint(config: ModelConfig) -> str:
    """Convenience wrapper to get the latest checkpoint."""
    return get_checkpoint(config, -1)


def get_checkpoint(config: ModelConfig, epoch: int, is_best: bool = False) -> str:
    """Get checkpoint path for specific epoch or latest if epoch=-1"""
    checkpoint_dir = Path(config.checkpoint_dir)

    if epoch == -1:
        # Get latest checkpoint when reloading from checkpoint
        model_files = list(Path(config.checkpoint_dir).glob("*.pth"))
        
        # Filter out best_model from sorting consideration
        regular_checkpoints = [f for f in model_files if not f.stem.startswith("best_model")]
        
        if not regular_checkpoints:
            return None
            
        # Sort only the regular checkpoints
        model_files = sorted(regular_checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
        return str(model_files[-1])
    else:
        # Get specific epoch checkpoint for saving models
        filename = config.model_name.format(
            epoch=epoch, arch=config.arch, dataset=config.dataset
        )
        
        if is_best:
            filename = "best_model.pth"
        
        checkpoint_path = checkpoint_dir / filename
        return str(checkpoint_path)

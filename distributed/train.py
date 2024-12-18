import argparse
import os
import warnings
import torch

from config import ModelConfig, get_default_config
from torch.distributed import destroy_process_group, init_process_group


def train_model(config: ModelConfig):
    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Avoid deadlocks when parsing the dataset in parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = get_default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_epochs", type=int, default=config.num_epochs)
    parser.add_argument("--lr", type=float, default=config.lr)

    args = parser.parse_args()

    # Update default configuration with command line arguments
    config.__dict__.update(vars(args))

    # Add local rank and global rank to the config
    config.local_rank = int(os.environ["LOCAL_RANK"])
    config.global_rank = int(os.environ["RANK"])

    assert config.local_rank != -1, "LOCAL_RANK environment variable not set"
    assert config.global_rank != -1, "RANK environment variable not set"

    # Print configuration once per node, where each node contains multiple GPUs
    if config.local_rank == 0:
        print("Configuration:")
        for key, value in config.__dict__.items():
            print(f"{key:>25}: {value}")

    # Setup distributed training
    init_process_group(backend="nccl")
    torch.cuda.set_device(config.local_rank)

    train_model(config)

    # Clean up distributed training
    destroy_process_group()

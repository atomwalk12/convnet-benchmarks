import argparse
import os
import shutil
import warnings
import torch

from config import ModelConfig, get_default_config
from torch.distributed import destroy_process_group, init_process_group

from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
import random
import torch.backends.cudnn as cudnn
import torchvision.models as models

from dataset import ImageNetDataset


def main():
    warnings.filterwarnings("ignore")

    # Avoid deadlocks when parsing the dataset in parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    config = get_default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_epochs", type=int, default=config.num_epochs)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--arch", type=str,
                        default="alexnet", choices=model_names)

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
            print(f"{key:>20}: {value}")

    # Setup distributed training
    init_process_group(backend="nccl")
    torch.cuda.set_device(config.local_rank)

    set_seed(config)
    main_worker(config)

    # Clean up distributed training
    destroy_process_group()


def set_seed(config: ModelConfig):
    # Set seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def main_worker(config: ModelConfig):
    # Setup the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {config.local_rank} - Using device: {device}")

    # Load dataset
    train_loader, val_loader = get_dataset(config)

    # Load model
    model = load_model(config)

    # Evaluate if flag is set
    if config.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(config.start_epoch, config.num_epochs):
        torch.cuda.empty_cache()
        
        train(train_loader, model, criterion, optimizer, epoch)
    
        
        if config.global_rank == 0:
            acc1 = validate(val_loader, model, criterion, args)
            

def get_dataset(config: ModelConfig):
    train_ds = load_dataset("Maysee/tiny-imagenet", split="train")
    val_ds = load_dataset("Maysee/tiny-imagenet", split="valid")

    train_ds = ImageNetDataset(train_ds)
    val_ds = ImageNetDataset(val_ds)

    train_dataloader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def train(train_loader, model, criterion, optimizer, epoch):
    pass


def validate(val_loader, model, criterion, args):
    pass


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_model(config: ModelConfig):
    model = models.__dict__[config.arch](pretrained=True)


if __name__ == "__main__":
    main()

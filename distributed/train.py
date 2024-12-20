import argparse
import os
import random
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
from config import ModelConfig, get_default_config
from dataset import ImageNetDataset
from datasets import load_dataset
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def main():
    warnings.filterwarnings("ignore")

    # Avoid deadlocks when parsing the dataset in parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    config = get_default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_epochs", type=int, default=config.num_epochs)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--arch", type=str, default="alexnet", choices=model_names)

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
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )


def main_worker(config: ModelConfig):
    # Setup the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {config.local_rank} - Using device: {device}")

    # Load dataset
    train_loader, val_loader = get_dataset(config)

    # Load training modules
    model = load_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1) # LR decays by 10% every 30 epochs

    model = DistributedDataParallel(model, device_ids=[config.local_rank])
    criterion = nn.CrossEntropyLoss().to(device)

    # TODO: Resume from checkpoint
    start_epoch = 0
    best_acc1 = 0
    
    # TODO: Define wandb logging

    # Evaluate if flag is set
    if config.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(start_epoch, config.num_epochs):
        torch.cuda.empty_cache()

        train(config, train_loader, model, criterion, optimizer, epoch, device)

        if config.global_rank == 0:
            acc1 = validate(val_loader, model, criterion)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if config.global_rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": config.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best,
            )


def get_dataset(config: ModelConfig):
    if config.dummy:
        train_ds = datasets.FakeData(1000, (3, 224, 224), 10, transforms.ToTensor())
        val_ds = datasets.FakeData(300, (3, 224, 224), 10, transforms.ToTensor())
    else:
        train_ds = load_dataset("Maysee/tiny-imagenet", split="train")
        val_ds = load_dataset("Maysee/tiny-imagenet", split="valid")

    train_ds = ImageNetDataset(train_ds)
    val_ds = ImageNetDataset(val_ds)

    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def train(config, train_loader, model, criterion, optimizer, epoch, device):
    model.train()

    batch_it = tqdm(
        train_loader, 
        desc=f"Processing Epoch {epoch:03d} on rank {config.global_rank}", 
        disable=config.local_rank != 0
    )

    for batch in batch_it:
        images, target = batch
        images = images.to(device)
        target = target.to(device)
        
        # Forward pass
        output = model(images)
        loss = criterion(output, target)

        # Compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar with current loss
        batch_it.set_postfix({
            'loss': f'{loss.item():6.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
        })


def validate(val_loader, model, criterion):
    
    return 0


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def load_model(config: ModelConfig, device: torch.device):
    model = models.__dict__[config.arch](pretrained=True)
    model.to(device)
    return model


if __name__ == "__main__":
    import sys
    sys.argv = [
        'train.py',  # Program name
        '--batch_size', '32',
        '--num_epochs', '10',
        '--lr', '0.001',
        '--arch', 'alexnet',
        '--num_epochs', '1',
    ]
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    main()

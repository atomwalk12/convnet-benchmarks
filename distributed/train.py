import argparse
import os
import random
import shutil
import time
import warnings
import wandb

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchmetrics
import torchvision.datasets as datasets
import torchvision.models as models
from config import ModelConfig, get_checkpoint, get_default_config, get_latest_checkpoint
from dataset import AverageMeter, ImageNetDataset
from datasets import load_dataset
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm


def main():
    warnings.filterwarnings("ignore")

    # Avoid deadlocks when parsing the dataset in parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )

    config = get_default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default=config.arch, choices=model_names)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_epochs", type=int, default=config.num_epochs)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--checkpoint_dir", type=str, default=config.checkpoint_dir)
    parser.add_argument("--model_name", type=str, default=config.model_name)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--evaluate", type=bool, default=config.evaluate)
    parser.add_argument("--dataset", type=str, default=config.dataset)
    parser.add_argument("--resume", type=str, default=config.resume)
    parser.add_argument("--exp_name", type=str, default=config.exp_group)
    parser.add_argument("--num_classes", type=int, default=config.num_classes)
    parser.add_argument(
        "--no_save_checkpoint", action="store_true", default=config.no_save_checkpoint
    )

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

    if torch.cuda.device_count() > 1:
        init_process_group(backend="nccl")
    else:
        # Use gloo backend for single GPU multi-process setup
        init_process_group(backend="gloo")

    # Force all processes to use GPU 0 when only one GPU is available
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank)
    else:
        torch.cuda.set_device(0)

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

    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Load dataset
    train_loader, val_loader = get_dataset(config)

    # Load training modules
    model = load_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)
    # LR decays by 10% every 30 epochs
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model, device_ids=[config.local_rank])
    else:
        # For single GPU, all processes use device 0
        model = DistributedDataParallel(model, device_ids=[0])

    criterion = nn.CrossEntropyLoss().to(device)

    # Resume from checkpoint
    start_epoch = 0
    best_acc1 = 0.0
    train_step = 0
    acc1 = 0.0
    wandb_run_id = None
    if config.resume:
        if config.resume == "latest":
            weights_path = get_latest_checkpoint(config)
        else:
            weights_path = get_checkpoint(config, int(config.resume))

        if weights_path and os.path.isfile(weights_path):
            print(f"Loading checkpoint from {weights_path}")
            checkpoint = torch.load(weights_path)

            # Here, I ensure that the 3 categories of parameters are loaded correctly
            # Training parameters
            start_epoch = checkpoint["last_epoch"] + 1
            best_acc1 = checkpoint["best_acc1"]
            train_step = checkpoint["global_train_step"] + 1

            # Model parameters
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])

            # Utils
            assert config.arch == checkpoint["arch"], "Model architecture mismatch"
            assert config.num_classes == checkpoint["num_classes"], "Number of classes mismatch"
            assert config.dataset == checkpoint["dataset"], "Dataset mismatch"
            wandb_run_id = checkpoint["wandb_run_id"]
        else:
            # Can happen if no prior sessions were started
            print(f"=> no checkpoint found, resuming from scratch...")

    # Only initialize W&B on the global rank 0 node
    if config.local_rank == 0:
        wandb.init(
            project="distributed_systems",
            name=f"global_rank_{config.global_rank}",
            id=wandb_run_id,
            resume="allow",
            group=config.exp_group,
            config=config,
            mode="offline",
        )

    # Evaluate if flag is set
    if config.evaluate:
        validate(config, val_loader, model, criterion, device, train_step)
        return

    for current_epoch in range(start_epoch, config.num_epochs):
        torch.cuda.empty_cache()

        train_step = train(
            config, train_loader, model, criterion, optimizer, current_epoch, device, train_step
        )

        if config.global_rank == 0:
            acc1 = validate(config, val_loader, model, criterion, device, train_step)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 < best_acc1  # Since we minimize the loss
        best_acc1 = min(acc1, best_acc1)

        if config.global_rank == 0 and not config.no_save_checkpoint:
            save_checkpoint(
                config,
                {
                    # State parameters
                    "last_epoch": current_epoch,
                    "best_acc1": best_acc1,
                    "global_train_step": train_step,
                    # Model parameters
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    # Utils
                    "arch": config.arch,
                    "num_classes": config.num_classes,
                    "dataset": config.dataset,
                    "wandb_run_id": wandb.run.id,
                },
                is_best,
                best_acc1,
            )


def get_dataset(config: ModelConfig):
    if config.dataset == "dummy":
        train_ds = datasets.FakeData(100, (3, 224, 224), config.num_classes, transforms.ToTensor())
        val_ds = datasets.FakeData(300, (3, 224, 224), config.num_classes, transforms.ToTensor())
    elif config.dataset == "tiny-imagenet":
        train_ds = load_dataset("Maysee/tiny-imagenet", split="train")
        val_ds = load_dataset("Maysee/tiny-imagenet", split="valid")
    else:
        raise ValueError(f"Dataset {config.dataset} not supported")

    train_ds = ImageNetDataset(train_ds)
    val_ds = ImageNetDataset(val_ds)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_ds, shuffle=True),
    )
    val_dataloader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def train(config, train_loader, model, criterion, optimizer, epoch, device, global_train_step):
    model.train()

    # Initialize variables to report metrics
    epoch_start_time = time.time()
    batch_times = []

    batch_it = tqdm(
        train_loader,
        desc=f"Processing Epoch {epoch:03d} on rank {config.global_rank}",
        disable=config.local_rank != 0,
    )

    for batch_idx, batch in enumerate(batch_it):
        batch_start_time = time.time()

        images, target = batch
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        # Compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure batch time
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        batch_times.append(batch_time)

        batch_it.set_postfix(
            {
                "loss": f"{loss.item():6.3f}",
                "lr": f'{optimizer.param_groups[0]["lr"]:.1e}',
                "batch_time": f"{batch_time:.3f}",
            }
        )

        # Log training loss every 100 batches
        if batch_idx % 3 == 0 and config.global_rank == 0:
            wandb.log({"train/batch_loss": loss.item()}, step=global_train_step)
        global_train_step += 1

    # Calculate metrics
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    throughput_per_second = len(train_loader) / epoch_time

    # Log for one node only
    if config.global_rank == 0:
        wandb.log(
            {
                "train/epoch_time": epoch_time,
                "train/avg_step_time": avg_batch_time,
                "train/throughput_per_second": throughput_per_second,
            },
            step=global_train_step,
        )
    return global_train_step


def validate(config, val_loader, model, criterion, device, global_train_step):
    model.eval()

    # Initialize metrics with sync_on_compute=False to prevent hanging
    metrics = {
        "val_loss": AverageMeter(sync_on_compute=False).to(device),
        "accuracy": torchmetrics.Accuracy(
            task="multiclass", num_classes=config.num_classes, sync_on_compute=False
        ).to(device),
        "precision": torchmetrics.Precision(
            task="multiclass", num_classes=config.num_classes, sync_on_compute=False
        ).to(device),
        "recall": torchmetrics.Recall(
            task="multiclass", num_classes=config.num_classes, sync_on_compute=False
        ).to(device),
        "f1": torchmetrics.F1Score(
            task="multiclass", num_classes=config.num_classes, sync_on_compute=False
        ).to(device),
    }

    with torch.no_grad():
        for batch in val_loader:
            images, target = batch
            images = images.to(device)
            target = target.to(device)

            # Forward pass
            output = model(images)  # [batch_size, num_classes]
            loss = criterion(output, target)  # scalar

            # Update metrics
            metrics["val_loss"].update(loss.item())
            preds = torch.argmax(output, dim=1)
            metrics["accuracy"].update(preds, target)
            metrics["precision"].update(preds, target)
            metrics["recall"].update(preds, target)
            metrics["f1"].update(preds, target)

    # Compute final metrics
    results = {
        "val_loss": metrics["val_loss"].compute(),
        "accuracy": metrics["accuracy"].compute(),
        "precision": metrics["precision"].compute(),
        "recall": metrics["recall"].compute(),
        "f1": metrics["f1"].compute(),
    }

    # Log metrics if using wandb
    if config.global_rank == 0:
        wandb.log({f"val/{k}": v for k, v in results.items()}, step=global_train_step, sync=False)

        print("\nValidation Results:")
        for k, v in results.items():
            print(f"{k:>10}: {v:.4f}")

    return results["val_loss"]


def save_checkpoint(config, state, is_best, acc1):
    # Obtain next checkpoint path
    checkpoint_path = get_checkpoint(config, state["last_epoch"])
    torch.save(state, checkpoint_path)

    # If this checkpoint is better, save it as the best current model
    if is_best:
        path = get_checkpoint(config, state["last_epoch"], is_best=True)
        print(f"Saving checkpoint {state['last_epoch']} (best - {acc1:.3f}) to {path}")
        shutil.copyfile(checkpoint_path, path)


def load_model(config: ModelConfig, device: torch.device):
    model = models.__dict__[config.arch](pretrained=True)
    model.to(device)
    return model


if __name__ == "__main__":
    import sys

    sys.argv = [
        "train.py",
        "--batch_size",
        "32",
        "--num_epochs",
        "33",
        "--lr",
        "0.001",
        "--arch",
        "alexnet",
        "--no_save_checkpoint",
    ]

    if "WORLD_SIZE" not in os.environ:
        print(f"World size is equal to {os.environ['WORLD_SIZE']}")
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    main()

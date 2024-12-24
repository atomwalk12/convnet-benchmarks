import torch
import torchmetrics
from datasets import Dataset


class ImageNetDataset:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class AverageMeter(torchmetrics.Metric):
    def __init__(self, sync_on_compute=False):
        super().__init__(sync_on_compute=sync_on_compute)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, value):
        self.sum += value
        self.count += 1

    def compute(self):
        return self.sum.float() / self.count

from config import ModelConfig
from datasets import load_dataset

class Dataset:
    def __init__(self, config: ModelConfig):
        self.config = config

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

# Dependencies
  - install nvidia-container-toolkit

# Setup
Run the following command to setup the environment:

```bash
# Start the containers
docker-compose -f docker-compose-multi-process.yaml up -d

# Enter the container and login to wandb
docker-compose -f docker-compose-multi-process.yaml exec node bash
wandb login

# Single node, simulate two GPUs using two processes
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=node:48123 train.py --no_checkpoint
```

For distributed training:

```bash
# Start the containers
docker-compose -f docker-compose-multi-node.yaml up -d

# Open terminal 1 and login to wandb:
docker-compose -f docker-compose-multi-node.yaml exec node0 bash
wandb login

# Open terminal 2 (already logging data to wandb on node0)
docker-compose -f docker-compose-multi-node.yaml exec node1 bash

# Run the training script in each terminal
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=node0:48123 train.py --no_checkpoint
```
    
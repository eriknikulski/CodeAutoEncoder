import os

import torch.distributed as dist
import torch.multiprocessing as mp

import const


def setup(rank, world_size):
    # TODO: choose open port; not hardcoded
    os.environ['MASTER_ADDR'] = const.MASTER_ADDR
    os.environ['MASTER_PORT'] = const.MASTER_PORT

    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(fn, world_size, train_dataloader, test_dataloader, experiment_name):
    mp.spawn(fn, args=(world_size, train_dataloader, test_dataloader, experiment_name,), nprocs=world_size, join=True)

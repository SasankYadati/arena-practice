import importlib
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim import AdamW

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


import part3_optimization.tests as tests
from part2_cnns.solutions import Linear, ResNet34

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

WORLD_SIZE = t.cuda.device_count()

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def get_cifar() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Returns CIFAR-10 train and test sets."""
    cifar_trainset = datasets.CIFAR10(exercises_dir / "data", train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(exercises_dir / "data", train=False, download=True, transform=IMAGENET_TRANSFORM)
    return cifar_trainset, cifar_testset

def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    if rank == 0:
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)
        print(f"{rank=}, received {received_tensor=}")
    dist.destroy_process_group()
    
def send_receive_nccl(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = t.device(f"cuda:{rank}")
    if rank == 0:
        sending_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        received_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)
        print(f"{rank=}, received {received_tensor=}")
    dist.destroy_process_group()

def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """
    if rank == src:
        for destination in range(world_size):
            if destination != rank:
                dist.send(tensor=tensor, dst=destination)
    else:
        dist.recv(tensor, src=0)

def reduce(tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces gradients to rank `dst`, so this process contains the sum or mean of all tensors across processes.
    """
    if rank != dst:
        dist.send(tensor=tensor, dst=dst)
    else:
        for other_rank in range(world_size):
            if other_rank != dst:
                received_tensor = t.zeros_like(tensor)
                dist.recv(received_tensor, src=other_rank)
                tensor += received_tensor
    if op == "mean":
        tensor /= world_size


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    reduce(tensor, rank, world_size, 0, op)
    broadcast(tensor, rank, world_size, 0)

def get_untrained_resnet(n_classes: int) -> ResNet34:
    """Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your implementation)."""
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


class DistResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")

    def pre_training_setup(self):
        self.model = get_untrained_resnet(self.args.n_classes).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr = self.args.learning_rate,
            weight_decay = self.args.weight_decay
        )
        self.trainset, self.testset = get_cifar()
        self.train_sampler = t.utils.data.DistributedSampler(
            self.trainset,
            num_replicas=self.args.world_size, # we'll divide each batch up into this many random sub-batches
            rank=self.rank, # this determines which sub-batch this process gets
        )
        self.train_loader = t.utils.data.DataLoader(
            self.trainset,
            self.args.batch_size, # this is the sub-batch size, i.e. the batch size that each GPU gets
            sampler=self.train_sampler,
            num_workers=2,  # setting this low so as not to risk bottlenecking CPU resources
            pin_memory=True,  # this can improve data transfer speed between CPU and GPU
        )
        self.test_sampler = t.utils.data.DistributedSampler(
            self.testset,
            num_replicas=self.args.world_size,
            rank=self.rank
        )
        self.test_loader = t.utils.data.DataLoader(
            self.testset,
            batch_size=self.args.batch_size, 
            sampler=self.test_sampler,
            num_workers=2,
            pin_memory=True
        )
        self.logged_variables = {"loss": [], "accuracy": []}
        self.examples_seen = 0
        for param in self.model.parameters():
            broadcast(param.data, self.rank, self.args.world_size, 0)
        if self.rank == 0:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        for param in self.model.parameters():
            all_reduce(param.grad, self.rank, self.args.world_size, "mean")
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.examples_seen += imgs.shape[0] * self.args.world_size
        if self.rank == 0:
            wandb.log({"loss": loss.item(), "rank": self.rank}, step=self.examples_seen)
        return loss
        

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating", disable=self.rank != 0):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum()
            total_samples += len(imgs)
        
        tensor = t.tensor([total_correct, total_samples], device=self.device)
        all_reduce(tensor, self.rank, self.args.world_size, "sum")
        accuracy = tensor[0] / tensor[1]
        if self.rank == 0:
            wandb.log({"accuracy": accuracy}, step=self.examples_seen)
        return accuracy
        

    def train(self):
        self.pre_training_setup()
        accuracy = self.evaluate()
        for epoch in range(self.args.epochs):
            self.model.train()
            self.train_sampler.set_epoch(epoch)
            self.test_sampler.set_epoch(epoch)
            pbar = tqdm(self.train_loader, desc="Training", disable=self.rank != 0)
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                loss = self.training_step(imgs, labels)
            accuracy = self.evaluate()
            if self.rank == 0:
                pbar.set_postfix(loss=f"{loss:.3f}", accuracy=f"{accuracy:.3f}", ex_seen=f"{self.examples_seen=:06}")
        if self.rank == 0:
            wandb.finish()
            t.save(self.model.state_dict(), f"resnet_{self.rank}.pth")


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


if MAIN:
    world_size = t.cuda.device_count()
    mp.spawn(dist_train_resnet_from_scratch, args=(world_size,), nprocs=world_size, join=True)

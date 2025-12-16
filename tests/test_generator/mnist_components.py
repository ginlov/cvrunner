import cvrunner
from time import time
from typing import Callable
from torch.utils.data import Dataset, DataLoader
from cvrunner.experiment.experiment import MetricType

import torch

class MnistDataset(Dataset):
    def __init__(self, train: bool = True, transform: Callable = None):
        from torchvision import datasets

        self.dataset = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"inputs": image, "labels": label}

class MnistClassifier(torch.nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MnistExperiment(cvrunner.experiment.BaseExperiment):
    def __init__(self) -> None:
        pass

    def runner_cls(self):
        return cvrunner.runner.distributed_train_runner.DistributedTrainRunner

    @property
    def batch_size(self) -> int:
        return 64

    @property
    def num_epochs(self) -> int:
        return 2

    @property
    def val_freq(self) -> int:
        return 1

    @property
    def wandb_project(self) -> str:
        return "mnist-cvrunner"

    @property
    def wandb_runname(self) -> None | str:
        return f"mnist_experiment_{int(time())}"
    
    def build_dataset(self, partition: str) -> Dataset:
        from torchvision import transforms

        transform = transforms.Compose([transforms.ToTensor()])

        if partition == "train":
            return MnistDataset(train=True, transform=transform)
        elif partition == "val":
            return MnistDataset(train=False, transform=transform)
        else:
            raise ValueError(f"Unknown partition: {partition}")

    def build_model(self) -> torch.nn.Module:
        return MnistClassifier()

    def build_dataloader(self, partition: str) -> DataLoader:
        # Build distributed dataloader
        dataset = self.build_dataset(partition)
        if partition == "train":
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=True,
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def build_optimizer_scheduler(
        self,
        model: torch.nn.Module
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return optimizer, scheduler
    
    def build_loss_function(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    # Build metric function
    def build_criterion(self, *args, **kwargs) -> Callable:
        # accuracy metric
        def accuracy(outputs, labels):
            _, preds = torch.max(outputs, dim=1)
            return (preds == labels).float().mean().item()
        return accuracy

    def train_step(
        self,
        model: torch.nn.Module,
        data_batch: dict,
        loss_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
    ) -> MetricType:
        model.train()
        inputs = data_batch["inputs"].to(device)
        labels = data_batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        return {"loss": loss.item()}

    def val_step(
        self,
        model: torch.nn.Module,
        data_batch: dict,
        loss_function: torch.nn.Module,
        criterion: Callable,
        device: torch.device,
    ) -> MetricType:
        model.eval()
        inputs = data_batch["inputs"].to(device)
        labels = data_batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            acc = criterion(outputs, labels)

        return {"val/loss": loss.detach(), "val/accuracy": acc}

    def load_checkpoint(self, file_path: str):
        pass
        return None
    
    def save_checkpoint(self) -> None:
        pass


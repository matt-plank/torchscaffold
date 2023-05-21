import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        training: DataLoader,
        validation: DataLoader,
        optimizer: optim.Optimizer,
        criterion,
        gpu_device=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.training = training
        self.validation = validation

        self.gpu_device = gpu_device

    def validation_metrics(self) -> tuple[float, float]:
        loss: float = 0
        accuracy: float = 0
        i: int = 0

        self.model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.validation):
                # Send inputs to GPU
                if self.gpu_device is not None:
                    inputs = inputs.to(self.gpu_device)
                    labels = labels.to(self.gpu_device)

                # Get predictions
                predictions = self.model(inputs)
                loss += self.criterion(predictions, labels).item()
                accuracy += (predictions.argmax(dim=1) == labels).float().mean().item()

        self.model.train()

        loss /= i + 1
        accuracy /= i + 1

        return loss, accuracy

    def fit(self, epochs: int, lr: float) -> dict:
        results = {
            "training_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }

        epoch_range = trange(epochs, desc="Epochs")
        for epoch in epoch_range:
            for i, (inputs, labels) in enumerate(self.training):
                # Send inputs to GPU
                if self.gpu_device is not None:
                    inputs = inputs.to(self.gpu_device)
                    labels = labels.to(self.gpu_device)

                # Gradient descent
                predictions = self.model(inputs)
                self.optimizer.zero_grad()
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()

                epoch_range.set_postfix(loss=loss.item())

            # Save metrics
            results["training_loss"].append(loss.item())
            validation_metrics = self.validation_metrics()
            results["validation_loss"].append(validation_metrics[0])
            results["validation_accuracy"].append(validation_metrics[1])

        return results

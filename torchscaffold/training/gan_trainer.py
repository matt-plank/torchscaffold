from typing import Protocol

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


class Generator(Protocol):
    """Protocol describing acceptable generator models."""

    def random(self, batch_size: int, gpu_device) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class GANTrainer:
    def __init__(
        self,
        generator: Generator,
        discriminator: nn.Module,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        criterion: nn.Module,
        training: DataLoader,
        batch_size: int,
        gpu_device,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.criterion = criterion

        self.training = training
        self.batch_size = batch_size

        self.gpu_device = gpu_device

    def update_discriminator(self, real_data) -> float:
        """Perform a single update of the discriminator - return the loss."""
        # Generate fake images
        _, fake_data = self.generator.random(self.batch_size, self.gpu_device)

        self.discriminator_optimizer.zero_grad()

        # Make predictions about known real images
        predictions_real = self.discriminator(real_data)
        all_ones = torch.ones_like(predictions_real, device=self.gpu_device)
        loss_for_real_data = self.criterion(predictions_real, all_ones)

        # Make predictions about known fake images
        predictions_fake = self.discriminator(fake_data.detach())
        all_zeros = torch.zeros_like(predictions_fake, device=self.gpu_device)
        loss_for_fake_data = self.criterion(predictions_fake, all_zeros)

        # Update the discriminator
        discriminator_loss = loss_for_real_data + loss_for_fake_data
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return discriminator_loss.item()

    def update_generator(self) -> float:
        """Perform a single update of the generator - return the loss."""
        self.generator_optimizer.zero_grad()
        _, fake_data = self.generator.random(self.batch_size, self.gpu_device)

        # Loss is based on how well the discriminator can identify fakes
        discriminator_predictions = self.discriminator(fake_data)
        all_ones = torch.ones_like(discriminator_predictions, device=self.gpu_device)
        loss = self.criterion(discriminator_predictions, all_ones)

        loss.backward()
        self.generator_optimizer.step()

        return loss.item()

    def fit_batch(self, real_data) -> tuple[float, float]:
        """Perform a single batch of training - return the generator and discriminator losses."""
        # Send inputs to GPU
        if self.gpu_device is not None:
            real_data = real_data.to(self.gpu_device)

        # Train the discriminator to recognise fakes
        discriminator_loss = self.update_discriminator(real_data)
        generator_loss = self.update_generator()

        return generator_loss, discriminator_loss

    def fit(self, epochs: int) -> dict:
        results = {
            "generator_loss": [],
            "discriminator_loss": [],
        }

        epoch_bar = trange(epochs, desc="Epochs")
        for epoch in epoch_bar:
            epoch_g_loss = 0
            epoch_d_loss = 0
            i = 0

            for i, (real_data, _) in tqdm(enumerate(self.training), total=len(self.training), leave=False):
                generator_loss, discriminator_loss = self.fit_batch(real_data)

                # Update progress bar
                epoch_d_loss += discriminator_loss
                epoch_g_loss += generator_loss

                epoch_bar.set_postfix(
                    g_loss=f"{epoch_g_loss / (i + 1):f}",
                    d_loss=f"{epoch_d_loss / (i + 1):.4f}",
                )

            results["generator_loss"].append(epoch_g_loss / (i + 1))
            results["discriminator_loss"].append(epoch_d_loss / (i + 1))

        return results

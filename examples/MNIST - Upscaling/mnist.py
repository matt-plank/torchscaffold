from pathlib import Path

import matplotlib.pyplot as plt
import torch
from config import config
from data import example_from_dataset, train_test_data
from model import UpscalingModel
from torch import nn
from torch.utils.data import DataLoader

from torchscaffold.training.model_trainer import ModelTrainer

GPU_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    """Entry point to the program. Either train a model or show some examples on the model."""
    args = config()

    train_data, test_data = train_test_data()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, shuffle=True)

    model = UpscalingModel()
    model.to(GPU_DEVICE)

    # Load model if it exists
    if Path(args.model_file).exists():
        model.load_state_dict(torch.load(args.model_file))

    if args.mode == "train":
        # Train the model
        trainer = ModelTrainer(
            model,
            train_loader,
            test_loader,
            torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            nn.MSELoss(),
            gpu_device=GPU_DEVICE,
        )

        trainer.fit(args.epochs, 0.001)

        # Save the model
        with open(args.model_file, "wb") as f:
            torch.save(model.state_dict(), f)

    if args.mode == "show":
        # Plot example with matplotlib
        example_input, example_output = example_from_dataset(test_data)
        example_input = example_input.to(GPU_DEVICE)
        example_output = example_output.to(GPU_DEVICE)

        prediction = model(example_input)
        prediction = prediction.cpu().detach().numpy().squeeze()

        fig, axes = plt.subplots(1, 3)

        for ax in axes:
            ax.axis("off")

        axes[0].imshow(example_input.cpu().detach().numpy().squeeze(), cmap="gray")
        axes[1].imshow(prediction, cmap="gray")
        axes[2].imshow(example_output.cpu().detach().numpy().squeeze(), cmap="gray")

        plt.show()


if __name__ == "__main__":
    main()

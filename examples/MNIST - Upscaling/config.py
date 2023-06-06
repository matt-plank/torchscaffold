import argparse


def config():
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("mode", choices=["train", "show"], help="Train or show model")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train for")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for training")
    parser.add_argument("--model-file", "-f", type=str, help="File to save model to / load model from")
    return parser.parse_args()

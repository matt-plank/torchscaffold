import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def draw_loss_on_axes(training_loss, validation_loss, axes: Axes) -> None:
    """Returns a figure of loss achieved during a training process."""
    axes.plot(training_loss, label="Training loss")
    axes.plot(validation_loss, label="Validation loss")
    axes.legend()
    axes.set_title("Loss")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")


def draw_accuracy_on_axes(accuracy, axes: Axes) -> None:
    """Returns a figure of accuracy achieved during a training process."""
    axes.plot(accuracy, label="Accuracy")
    axes.legend()
    axes.set_title("Accuracy")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Accuracy")

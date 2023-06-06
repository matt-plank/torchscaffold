from torchvision import datasets, transforms

transform_inputs = transforms.Compose(
    [
        transforms.Resize((7, 7)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


transform_outputs = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


class MNISTUpscalingDataset(datasets.MNIST):
    """Modified MNIST dataset for upscaling problem."""

    def __getitem__(self, index):
        """Gets handwritten digit, returns low-res and normal-res versions."""
        img, _ = super().__getitem__(index)
        return transform_inputs(img), transform_outputs(img)


def train_test_data():
    """Returns the train and test data."""
    train_data = MNISTUpscalingDataset(root="~/.pytorch/MNIST_data", train=True, download=True)
    test_data = MNISTUpscalingDataset(root="~/.pytorch/MNIST_data", train=False, download=True)

    return train_data, test_data


def example_from_dataset(dataset):
    """Return input and output from an example in a dataset."""
    return dataset[0][0].unsqueeze(0), dataset[0][1].unsqueeze(0)

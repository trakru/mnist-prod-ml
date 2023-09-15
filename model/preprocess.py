from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=64, train=True, val=True):
    """Function to get data loaders for the MNIST dataset.

    Args:
        batch_size (int, optional): Batch size for the data loader. Defaults to 64.
        train (bool, optional): Whether to get the training set data loader. Defaults to True.
        val (bool, optional): Whether to get the validation set data loader. Defaults to True.

    Returns:
        tuple: Data loaders for the training and validation sets.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))] # mean and std of MNIST dataset
    )

    # NOTE: Currently, the dataset gets downloaded every time this script runs. 
    # It would be more efficient to implement a check to see if the dataset 
    # already exists locally before downloading it again (using python `hashlib`)

    # Loading the MNIST dataset
    train_dataset = (
        datasets.MNIST("./data", train=True, download=True, transform=transform)
        if train
        else None
    )
    val_dataset = (
        datasets.MNIST("./data", train=False, download=True, transform=transform)
        if val
        else None
    )

    train_loader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if train
        else None
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val else None
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()

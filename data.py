import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

    CIFAR100 = CIFAR10

    class FashionMNIST:

        class VGG:

            train = transforms.Compose([
                transforms.Pad(2),  # 28x28 -> 32x32
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Grayscale(num_output_channels=3),  # Convert 1->3 channels
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.2860, 0.2860, 0.2860],  # F-MNIST stats replicated 3x
                    std=[0.3530, 0.3530, 0.3530]
                )
            ])

            test = transforms.Compose([
                transforms.Pad(2),  # 28x28 -> 32x32
                transforms.Grayscale(num_output_channels=3),  # Convert 1->3 channels
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.2860, 0.2860, 0.2860],
                    std=[0.3530, 0.3530, 0.3530]
                )
            ])


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True, split_test_from_train=False):
    # MPS doesn't work well with multiprocessing, use 0 workers
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        num_workers = 0

    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.test)

    # Load original test set
    original_test_set = ds(path, train=False, download=True, transform=transform.test)

    # Get dataset sizes
    train_size = len(train_set.data)
    test_size = len(original_test_set.data)
    total_size = train_size + test_size

    print(f'Combining {dataset}: train ({train_size}) + test ({test_size}) = {total_size} total, then shuffling and splitting back')

    # Combine train + test
    combined_data = np.concatenate([train_set.data, original_test_set.data], axis=0)

    # Handle different target types (list for CIFAR, tensor for FashionMNIST)
    if isinstance(train_set.targets, torch.Tensor):
        # FashionMNIST case: targets are tensors
        combined_targets = torch.cat([train_set.targets, original_test_set.targets], dim=0)
        combined_targets = combined_targets.numpy()
    else:
        # CIFAR case: targets are lists
        combined_targets = train_set.targets + original_test_set.targets
        combined_targets = np.array(combined_targets)

    # Shuffle with fixed seed for reproducibility
    indices = np.arange(total_size)
    np.random.seed(412)  # Fixed seed for reproducibility
    np.random.shuffle(indices)

    combined_data = combined_data[indices]
    combined_targets = combined_targets[indices]

    # Split back to original proportions
    train_set.data = combined_data[:train_size]
    train_set.targets = combined_targets[:train_size].tolist()

    test_set = ds(path, train=False, download=True, transform=transform.test)
    test_set.data = combined_data[train_size:total_size]
    test_set.targets = combined_targets[train_size:total_size].tolist()

    val_set = None


    # pin_memory not supported on MPS
    pin_mem = not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())

    loaders_dict = {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_mem
        ),
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem
        ),
    }

    # Add validation loader if 3-way split is used
    if val_set is not None:
        loaders_dict['val'] = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem
        )

    return loaders_dict, max(train_set.targets) + 1

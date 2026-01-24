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
    test_set = ds(path, train=False, download=True, transform=transform.test)

    # If use_test is True, shuffle the order within each split (train stays train, test stays test)
    if use_test:
        print(f'Shuffling {dataset}: train ({len(train_set.data)}) and test ({len(test_set.data)}) splits independently')

        # Store original data type (tensor for FashionMNIST, numpy for CIFAR)
        train_is_tensor = isinstance(train_set.data, torch.Tensor)
        test_is_tensor = isinstance(test_set.data, torch.Tensor)

        # Shuffle train set
        train_size = len(train_set.data)
        train_indices = np.arange(train_size)
        np.random.seed(412)  # Fixed seed for reproducibility
        np.random.shuffle(train_indices)

        if train_is_tensor:
            train_set.data = train_set.data[train_indices]
            train_set.targets = torch.tensor(train_set.targets)[train_indices].tolist()
        else:
            train_set.data = train_set.data[train_indices]
            if isinstance(train_set.targets, list):
                train_set.targets = [train_set.targets[i] for i in train_indices]
            else:
                train_set.targets = np.array(train_set.targets)[train_indices].tolist()

        # Shuffle test set
        test_size = len(test_set.data)
        test_indices = np.arange(test_size)
        np.random.seed(413)  # Different seed for test set
        np.random.shuffle(test_indices)

        if test_is_tensor:
            test_set.data = test_set.data[test_indices]
            test_set.targets = torch.tensor(test_set.targets)[test_indices].tolist()
        else:
            test_set.data = test_set.data[test_indices]
            if isinstance(test_set.targets, list):
                test_set.targets = [test_set.targets[i] for i in test_indices]
            else:
                test_set.targets = np.array(test_set.targets)[test_indices].tolist()

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

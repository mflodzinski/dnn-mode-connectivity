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
    train_set = ds(path, train=True, download=True, transform=transform.train)

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.train)
        val_set = None
    else:
        if split_test_from_train:
            # 3-way split: train (40K) + val (5K) + test (5K) from training data
            print("Using train (40000) + validation (5000) + test (5000)")
            full_train_data = train_set.data
            full_train_targets = train_set.targets

            # Train: indices 0-39999
            train_set.data = full_train_data[:40000]
            train_set.targets = full_train_targets[:40000]

            # Validation: indices 40000-44999
            val_set = ds(path, train=True, download=True, transform=transform.test)
            val_set.train = False
            val_set.data = full_train_data[40000:45000]
            val_set.targets = full_train_targets[40000:45000]

            # Test: indices 45000-49999
            test_set = ds(path, train=True, download=True, transform=transform.test)
            test_set.train = False
            test_set.data = full_train_data[45000:50000]
            test_set.targets = full_train_targets[45000:50000]
        else:
            # 2-way split: train (45K) + validation (5K) - original behavior but fixed
            print("Using train (45000) + validation (5000)")
            full_train_data = train_set.data
            full_train_targets = train_set.targets

            # Train: indices 0-44999
            train_set.data = full_train_data[:45000]
            train_set.targets = full_train_targets[:45000]

            # Validation: indices 45000-49999
            test_set = ds(path, train=True, download=True, transform=transform.test)
            test_set.train = False
            test_set.data = full_train_data[45000:50000]
            test_set.targets = full_train_targets[45000:50000]
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

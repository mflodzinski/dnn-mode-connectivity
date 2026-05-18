import os
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

def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True, split_test_from_train=False, eval_mode=False):
    # MPS doesn't work well with multiprocessing, use 0 workers
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        num_workers = 0

    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    # Use test transform (no augmentation) for train set during evaluation
    train_transform = transform.test if eval_mode else transform.train
    train_set = ds(path, train=True, download=True, transform=train_transform)

    print('You are going to run models on the test set. Are you sure?')
    test_set = ds(path, train=False, download=True, transform=transform.test)
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

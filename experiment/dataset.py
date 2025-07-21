import os
import numpy as np
import torch
from torchvision import datasets, transforms
from experiment.tinyimagenet import MyTinyImagenet
from torch.utils.data import TensorDataset
import random


def get_data(dataset_name, batch_size, n_workers):
    if "cifar" in dataset_name:
        return get_cifar_data(dataset_name, batch_size, n_workers)
    elif dataset_name == "tiny_imagenet":
        return get_tinyimagenet(batch_size, n_workers)
    else:
        raise Exception('unknown dataset!')

class_distribution_table_cifar100LT = [21, 36, 13, 344, 171, 10, 7, 24, 15, 14, 77, 7, 434, 6, 38, 328, 149, 12, 67, 85, 33, 19, 13, 477, 
                            9, 206, 226, 48, 135, 42, 273, 11, 61, 11, 378, 32, 10, 237, 248, 64, 7, 74, 17, 30, 12, 44, 197, 
                            314, 118, 40, 89, 6, 260, 18, 5, 5, 5, 455, 25, 23, 70, 179, 98, 9, 163, 102, 8, 188, 5, 500, 8, 
                            142, 216, 6, 299, 286, 56, 156, 123, 58, 27, 20, 93, 29, 361, 26, 15, 396, 112, 415, 46, 53, 16, 
                            6, 81, 22, 129, 51, 35, 107]
def get_cifar_data(dataset_name, batch_size, n_workers):
    data = {}
    size = [3, 32, 32]
    if dataset_name == "cifar10":
        task_num = 5
        class_num = 10
        data_dir = './data/binary_cifar_/'
    elif dataset_name == "cifar100":
        task_num = 10
        class_num = 100
        data_dir = './data/binary_cifar100_10/'
    class_per_task = class_num // task_num

    if dataset_name == "cifar100":
        random.shuffle(class_distribution_table_cifar100LT)
    #     n_shuffle = random.randint(1, 10)
    #     for _ in range(n_shuffle):
    #         random.shuffle(class_distribution_table_cifar100LT)
        #print(f"Shuffled class_distribution_table_cifar100LT {n_shuffle} times.")
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        dataset_path = './data/'
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = {}
        if dataset_name == "cifar10":
            dataset['train'] = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        elif dataset_name == "cifar100" or dataset_name == "cifar100_50":

            dataset['train'] = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for task_id in range(task_num):
            data[task_id] = {}
            for data_type in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
                data[task_id][data_type] = {'x': [], 'y': []}
                # Nếu là train và cifar100, lấy theo phân phối LT
                if dataset_name == "cifar100" and data_type == 'train':
                    # Đếm số lượng đã lấy cho từng class
                    class_counts = [0 for _ in range(100)]
                    for image, target in loader:
                        label = target.numpy()[0]
                        if label in range(class_per_task * task_id, class_per_task * (task_id + 1)):
                            if class_counts[label] < class_distribution_table_cifar100LT[label]:
                                data[task_id][data_type]['x'].append(image)
                                data[task_id][data_type]['y'].append(label)
                                class_counts[label] += 1
                else:
                    for image, target in loader:
                        label = target.numpy()[0]
                        if label in range(class_per_task * task_id, class_per_task * (task_id + 1)):
                            data[task_id][data_type]['x'].append(image)
                            data[task_id][data_type]['y'].append(label)

        # save
        for task_id in data.keys():
            for data_type in ['train', 'test']:
                data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x']).view(-1, size[0], size[1], size[2])
                data[task_id][data_type]['y'] = torch.LongTensor(np.array(data[task_id][data_type]['y'], dtype=int)).view(-1)
                torch.save(data[task_id][data_type]['x'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'x.bin'))
                torch.save(data[task_id][data_type]['y'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(task_num))
    print('Task order =', ids)
    for i in range(task_num):
        data[i] = dict.fromkeys(['train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])

        dataset_new_train = torch.utils.data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = torch.utils.data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            #to test DELTA, we use 16 for train batch size (in parameter: --batch_size 16)
            #batch_size=16,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            #to test DELTA, we use 128 for test batch size
            #batch_size=128,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader
    

    # Print class distribution and total images after everything is set up
    if dataset_name == "cifar100":
        train_class_counts = [0 for _ in range(class_num)]
        test_class_counts = [0 for _ in range(class_num)]
        for t in range(task_num):
            y_train = data[t]['train']['y']
            y_test = data[t]['test']['y']
            if hasattr(y_train, 'numpy'):
                y_train = y_train.numpy()
            if hasattr(y_test, 'numpy'):
                y_test = y_test.numpy()
            for c in range(class_num):
                train_class_counts[c] += int((y_train == c).sum())
                test_class_counts[c] += int((y_test == c).sum())
    
        print("Train distribution: ")
        print(class_distribution_table_cifar100LT)
        print("Train class distribution:")
        print(train_class_counts)
        print(f"Total train images: {sum(train_class_counts)}")
        print("Test class distribution:")
        print(test_class_counts)
        print(f"Total test images: {sum(test_class_counts)}")
    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size


def get_tinyimagenet(batch_size, n_workers):
    data = {}
    size = [3, 64, 64]
    task_num = 100
    class_num = 200
    class_per_task = class_num // task_num

    base_path = './data/TINYIMG'
    data_dir = './data/binary_tiny200_100'

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        dat = {}
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train = MyTinyImagenet(base_path, train=True, download=True, transform=test_transform)
        test = MyTinyImagenet(base_path, train=False, download=True, transform=test_transform)

        dat['train'] = train
        dat['test'] = test
        for t in range(task_num):
            data[t] = {}
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(class_per_task * t, class_per_task * (t + 1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)

        # and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(task_num))
    print('Task order =', ids)
    for i in range(task_num):
        data[i] = dict.fromkeys(['train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])

        dataset_new_train = torch.utils.data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = torch.utils.data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader

    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size

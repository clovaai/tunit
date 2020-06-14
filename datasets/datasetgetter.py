"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import torch
from torchvision.datasets import ImageFolder
import os
import torchvision.transforms as transforms
from datasets.custom_dataset import ImageFolerRemap, CrossdomainFolder


class DuplicatedCompose(object):
    def __init__(self, tf1, tf2):
        self.tf1 = tf1
        self.tf2 = tf2

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t1 in self.tf1:
            img1 = t1(img1)
        for t2 in self.tf2:
            img2 = t2(img2)
        return img1, img2


def get_dataset(dataset, args):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = DuplicatedCompose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   normalize],
                                  [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                   transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1),
                                                                ratio=(0.9, 1.1), interpolation=2),
                                   transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), normalize])

    transform_val = DuplicatedCompose([transforms.Resize((args.img_size, args.img_size)),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1),
                                                                    ratio=(0.9, 1.1), interpolation=2),
                                       transforms.ToTensor(), normalize])

    class_to_use = args.att_to_use

    print('USE CLASSES', class_to_use)

    # remap labels
    remap_table = {}
    i = 0
    for k in class_to_use:
        remap_table[k] = i
        i += 1

    print("LABEL MAP:", remap_table)

    if 'afhq' in dataset.lower():
        print('USE AFHQ dataset [FOR IIC]')
        print("LABEL MAP:", remap_table)

        img_train_dir = os.path.join(args.data_dir, 'afhq', 'train')
        img_test_dir = os.path.join(args.data_dir, 'afhq', 'test')

        train_dataset = ImageFolerRemap(img_train_dir, transform=transform, remap_table=remap_table)
        train_with_idx = ImageFolerRemap(img_train_dir, transform=transform, remap_table=remap_table, with_idx=True)
        val_dataset = ImageFolerRemap(img_test_dir, transform=transform_val, remap_table=remap_table)

        tot_targets = torch.tensor(train_dataset.targets)
        val_targets = torch.tensor(val_dataset.targets)

        train_idx = None
        val_idx = None

        min_data = 99999999
        max_data = 0

        for k in class_to_use:
            tmp_idx = (tot_targets == k).nonzero()
            tmp_val_idx = (val_targets == k).nonzero()

            tot_train_tmp = len(tmp_idx)

            if min_data > tot_train_tmp:
                min_data = tot_train_tmp
            if max_data < tot_train_tmp:
                max_data = tot_train_tmp

            if k == class_to_use[0]:
                train_idx = tmp_idx.clone()
                val_idx = tmp_val_idx.clone()
            else:
                train_idx = torch.cat((train_idx, tmp_idx))
                val_idx = torch.cat((val_idx, tmp_val_idx))

        train_dataset_ = torch.utils.data.Subset(train_dataset, train_idx)
        val_dataset_ = torch.utils.data.Subset(val_dataset, val_idx)

        args.min_data = min_data
        args.max_data = max_data

        train_dataset = {'TRAIN': train_dataset_, 'FULL': train_dataset, 'IDX': train_with_idx}
        val_dataset = {'VAL': val_dataset_, 'FULL': val_dataset}

    elif dataset.lower() == 'animal_faces':
        print('USE Animals ImageNet Subset dataset [WITH IIC]')
        print("LABEL MAP:", remap_table)

        img_dir = os.path.join(args.data_dir, 'animal_faces')

        # divide into training and validation set
        # divide into labeled and unlabeled set
        if 0.0 < args.p_semi < 1.0:
            assert 'SEMI' in args.train_mode
            dataset_sup = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table)
            dataset_un = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table)
            valdataset = ImageFolerRemap(img_dir, transform=transform_val, remap_table=remap_table)
            # parse classes to use
            tot_targets = torch.tensor(dataset_sup.targets)

            train_sup_idx = None
            train_unsup_idx = None
            val_idx = None
            min_data = 99999999
            max_data = 0
            for k in class_to_use:
                tmp_idx = (tot_targets == k).nonzero()
                train_tmp_idx = tmp_idx[:-50]
                val_tmp_idx = tmp_idx[-50:]

                tot_train_tmp = len(train_tmp_idx)
                num_sup = int(tot_train_tmp * args.p_semi)

                if min_data > tot_train_tmp:
                    min_data = tot_train_tmp
                if max_data < tot_train_tmp:
                    max_data = tot_train_tmp

                if num_sup == 0:
                    num_sup = 1

                train_tmp_sup = train_tmp_idx[:num_sup]
                train_tmp_unsup = train_tmp_idx[num_sup:]
                print("FOR CLASS[{}] SUP[{}] UNSUP[{}]".format(k, len(train_tmp_sup), len(train_tmp_unsup)))
                if k == class_to_use[0]:
                    train_sup_idx = train_tmp_sup.clone()
                    train_unsup_idx = train_tmp_unsup.clone()
                    val_idx = val_tmp_idx.clone()
                else:
                    train_sup_idx = torch.cat((train_sup_idx, train_tmp_sup))
                    train_unsup_idx = torch.cat((train_unsup_idx, train_tmp_unsup))
                    val_idx = torch.cat((val_idx, val_tmp_idx))

            args.min_data = min_data
            args.max_data = max_data
            print("MINIMUM DATA :", args.min_data)
            print("MAXIMUM DATA :", args.max_data)

            train_sup_dataset = torch.utils.data.Subset(dataset_sup, train_sup_idx)
            train_unsup_dataset = torch.utils.data.Subset(dataset_un, train_unsup_idx)
            val_dataset = torch.utils.data.Subset(valdataset, val_idx)

            train_dataset = {'SUP': train_sup_dataset, 'UNSUP': train_unsup_dataset, 'FULL': dataset_sup}
        else:
            assert (args.p_semi == 0.0 or args.p_semi == 1.0)
            dataset = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table)
            valdataset = ImageFolerRemap(img_dir, transform=transform_val, remap_table=remap_table)
            # parse classes to use
            tot_targets = torch.tensor(dataset.targets)

            min_data = 99999999
            max_data = 0

            train_idx = None
            val_idx = None
            for k in class_to_use:
                tmp_idx = (tot_targets == k).nonzero()
                train_tmp_idx = tmp_idx[:-50]
                val_tmp_idx = tmp_idx[-50:]
                if k == class_to_use[0]:
                    train_idx = train_tmp_idx.clone()
                    val_idx = val_tmp_idx.clone()
                else:
                    train_idx = torch.cat((train_idx, train_tmp_idx))
                    val_idx = torch.cat((val_idx, val_tmp_idx))
                if min_data > len(train_tmp_idx):
                    min_data = len(train_tmp_idx)
                if max_data < len(train_tmp_idx):
                    max_data = len(train_tmp_idx)

            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(valdataset, val_idx)

            args.min_data = min_data
            args.max_data = max_data
            print("MINIMUM DATA :", args.min_data)
            print("MAXIMUM DATA :", args.max_data)

            train_dataset = {'TRAIN': train_dataset, 'FULL': dataset}

    elif '2' in dataset.lower():
        print('USE {}'.format(dataset.lower()))

        data_to_use = dataset.lower().split('2')

        dataset_prefix = 'art5' if 'photo2' in dataset.lower() else dataset.lower()

        for i in range(2):
            for mode in ['train', 'test']:
                is_exist_dir = os.path.join(args.data_dir, dataset_prefix, mode, data_to_use[i])
                if not os.path.exists(is_exist_dir):
                    print("Directory {} does not exist".format(is_exist_dir))
                    exit(-1)

        img_train_dir = os.path.join(args.data_dir, dataset_prefix, 'train')
        img_test_dir = os.path.join(args.data_dir, dataset_prefix, 'test')

        transform = DuplicatedCompose([transforms.Resize((args.img_size, args.img_size)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1),
                                                                    ratio=(0.9, 1.1), interpolation=2),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), normalize])

        # divide into training and validation set
        # divide into labeled and unlabeled set
        if 0.0 < args.p_semi < 1.0:
            assert 'SEMI' in args.train_mode
            dataset_sup = CrossdomainFolder(img_train_dir, data_to_use, transform)
            dataset_un = CrossdomainFolder(img_train_dir, data_to_use, transform)
            val_dataset = CrossdomainFolder(img_test_dir, data_to_use, transform_val)

            # parse classes to use
            tot_targets = torch.tensor(dataset_sup.targets)

            # num data
            train_targets = torch.tensor(dataset_sup.targets)
            val_targets = torch.tensor(val_dataset.targets)
            min_data = 99999999
            max_data = 0
            for k in class_to_use:
                tmp_idx = (train_targets == k).nonzero()
                val_tmp_idx = (val_targets == k).nonzero()
                if min_data > len(tmp_idx):
                    min_data = len(tmp_idx)
                if max_data < len(tmp_idx):
                    max_data = len(tmp_idx)
                print(k, len(tmp_idx), len(val_tmp_idx))
            args.min_data = min_data
            args.max_data = max_data
            print("MINIMUM DATA :", args.min_data)
            print("MAXIMUM DATA :", args.max_data)

            train_sup_idx = None
            train_unsup_idx = None
            for k in [0, 1]:
                tmp_idx = (tot_targets == k).nonzero()

                tot_train_tmp = len(tmp_idx)
                num_sup = int(tot_train_tmp * args.p_semi)

                if num_sup == 0:
                    num_sup = 1

                train_tmp_sup = tmp_idx[:num_sup]
                train_tmp_unsup = tmp_idx[num_sup:]
                print("FOR CLASS[{}] SUP[{}] UNSUP[{}]".format(k, len(train_tmp_sup), len(train_tmp_unsup)))
                if k == 0:
                    train_sup_idx = train_tmp_sup.clone()
                    train_unsup_idx = train_tmp_unsup.clone()
                else:
                    train_sup_idx = torch.cat((train_sup_idx, train_tmp_sup))
                    train_unsup_idx = torch.cat((train_unsup_idx, train_tmp_unsup))

            train_sup_dataset = torch.utils.data.Subset(dataset_sup, train_sup_idx)
            train_unsup_dataset = torch.utils.data.Subset(dataset_un, train_unsup_idx)

            train_dataset = {'SUP': train_sup_dataset, 'UNSUP': train_unsup_dataset, 'FULL': dataset_sup}
        else:
            assert (args.p_semi == 0.0 or args.p_semi == 1.0)
            train_dataset = CrossdomainFolder(img_train_dir, data_to_use, transform)
            val_dataset = CrossdomainFolder(img_test_dir, data_to_use, transform_val)

            # num data
            train_targets = torch.tensor(train_dataset.targets)
            val_targets = torch.tensor(val_dataset.targets)
            min_data = 999999999
            max_data = 0
            for k in [0, 1]:
                tmp_idx = (train_targets == k).nonzero()
                val_tmp_idx = (val_targets == k).nonzero()
                if min_data > len(tmp_idx):
                    min_data = len(tmp_idx)
                if max_data < len(tmp_idx):
                    max_data = len(tmp_idx)
                print(k, len(tmp_idx), len(val_tmp_idx))
            args.min_data = min_data
            args.max_data = max_data
            print("MINIMUM DATA :", args.min_data)
            print("MAXIMUM DATA :", args.max_data)

            train_dataset = {'TRAIN': train_dataset, 'FULL': train_dataset}

    elif dataset.lower() in ['lsun_car', 'ffhq']:
        num_val = -10000 if dataset.lower() == 'lsun_car' else -7000
        print("USE LSUN CAR / FFHQ DATASET :", num_val)
        print(args.data_dir)
        sub_root = 'lsun-car' if dataset.lower() == 'lsun_car' else 'ffhq'
        img_dir = os.path.join(args.data_dir, sub_root)

        dataset = ImageFolder(img_dir, transform=transform)
        valdataset = ImageFolder(img_dir, transform=transform_val)

        tot_targets = torch.tensor(dataset.targets)

        tmp_idx = (tot_targets == 0).nonzero()
        train_idx = tmp_idx[:num_val].clone()
        val_idx = tmp_idx[num_val:].clone()

        min_data = len(train_idx)
        max_data = len(train_idx)

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(valdataset, val_idx)

        args.min_data = min_data
        args.max_data = max_data
        print("MINIMUM DATA :", args.min_data)
        print("MAXIMUM DATA :", args.max_data)

        train_dataset = {'TRAIN': train_dataset, 'FULL': dataset}

    else:
        print('NOT IMPLEMENTED DATASET :', dataset)
        exit(-3)

    return train_dataset, val_dataset



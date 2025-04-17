from collections import Counter
import os
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import pytorch_lightning as pl

from data_loader.flist_dataset import ImageFolder

class TransferDataModule(pl.LightningDataModule):
    def __init__(self, source_path, target_path, test_path, args) -> None:
        super().__init__()
        self.src_path = source_path
        self.tgt_path = target_path
        self.test_path = test_path

        self.args = args
        self.batch_size= args.batch_size
        self.num_workers = args.num_workers

    def get_transforms(self):
        source_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        target_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return source_transforms, target_transforms, test_transforms

    def setup(self, stage):
        transform_pack = self.get_transforms()
        src_trans, tgt_trans, test_trans = transform_pack

        if stage == 'fit' or stage is None:
            self.source_dataset = ImageFolder(self.src_path, transform=src_trans, train=True, return_id=False)
            self.target_dataset = ImageFolder(self.tgt_path, transform=tgt_trans, train=True, return_id=False)
            self.val_dataset = ImageFolder(self.test_path, transform=test_trans, train=False, return_paths=False)
            freq = Counter(self.source_dataset.labels)
            class_weight = {x: 1.0 / freq[x] for x in freq}
            source_weights = [class_weight[x] for x in self.source_dataset.labels]
            self.sampler = WeightedRandomSampler(source_weights,
                                            len(self.source_dataset.labels))
        if stage == 'test' or stage is None:
            self.test_dataset = ImageFolder(self.test_path, transform=test_trans, train=False, return_paths=True)
            self.sampler = None

    def prepare_data(self):
        pass

    def train_dataloader(self):
        source_loader = DataLoader(self.source_dataset, batch_size=self.batch_size, sampler=self.sampler, drop_last=True, num_workers=self.num_workers)
        target_loader = DataLoader(self.target_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        return {"src": source_loader, "tgt": target_loader}

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=192, shuffle=False, num_workers=self.num_workers)


class TransferCLIPDataModule(pl.LightningDataModule):
    def __init__(self, source_path, target_path, test_path, args) -> None:
        super().__init__()
        self.src_path = source_path
        self.tgt_path = target_path
        self.test_path = test_path

        self.args = args
        self.batch_size= args.batch_size
        self.num_workers = args.num_workers

    def get_transforms(self):
        source_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        target_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        CLIP_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return source_transforms, target_transforms, CLIP_transforms, test_transforms

    def setup(self, stage):
        transform_pack = self.get_transforms()
        src_trans, tgt_trans, clip_trans, test_trans = transform_pack

        if stage == 'fit' or stage is None:
            self.source_dataset = ImageFolder(self.src_path, transform=[src_trans, clip_trans], train=True, return_id=False)
            self.target_dataset = ImageFolder(self.tgt_path, transform=[tgt_trans, clip_trans], train=True, return_id=False)
            self.val_dataset = ImageFolder(self.test_path, transform=test_trans, train=False, return_paths=False)
            freq = Counter(self.source_dataset.labels)
            class_weight = {x: 1.0 / freq[x] for x in freq}
            source_weights = [class_weight[x] for x in self.source_dataset.labels]
            self.sampler = WeightedRandomSampler(source_weights,
                                            len(self.source_dataset.labels))
        if stage == 'test' or stage is None:
            self.test_dataset = ImageFolder(self.test_path, transform=test_trans, train=False)#, return_paths=True)
            self.sampler = None

    def prepare_data(self):
        pass

    def train_dataloader(self):
        source_loader = DataLoader(self.source_dataset, batch_size=self.batch_size, sampler=self.sampler, drop_last=True, num_workers=self.num_workers)
        target_loader = DataLoader(self.target_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        return {"src": source_loader, "tgt": target_loader}

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=192, shuffle=False, num_workers=self.num_workers)

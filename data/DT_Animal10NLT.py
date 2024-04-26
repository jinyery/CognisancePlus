######################################
#         Jinyery Yang
######################################


import os
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from randaugment import RandAugment


class Animal10NLT(data.Dataset):
    def __init__(
        self, phase, data_path, rgb_mean, rgb_std, rand_aug, output_path, logger
    ):
        super(Animal10NLT, self).__init__()
        valid_phase = ["train", "val", "test"]
        assert phase in valid_phase
        if phase == "train":
            full_phase = "train"
        else:
            full_phase = "test"
        logger.info("====== The Current Split is : {}".format(full_phase))
        if "~" in data_path:
            data_path = os.path.expanduser(data_path)
        # if "~" in anno_path:
        #     anno_path = os.path.expanduser(anno_path)
        # logger.info('====== The data_path is : {}, the anno_path is {}.'.format(data_path, anno_path))
        self.logger = logger

        self.phase = phase
        self.rand_aug = rand_aug
        self.data_path = data_path

        self.transform = self.get_data_transform(phase, rgb_mean, rgb_std)

        # load all image info
        logger.info("=====> Load image info")
        self.img_paths, self.labels = self.load_img_info()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_paths[index]
        label = self.labels[index]

        path = os.path.basename(path)
        if self.phase == "train":
            img_path = os.path.join(self.data_path, "training_selected2")
        else:
            img_path = os.path.join(self.data_path, "testing")
        with open(os.path.join(img_path, path), "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        # It has no practical significance, just to align with other formats.
        align = {}
        if self.phase != "train":
            return sample, label, align, align, index
        else:
            return sample, label, align, index

    #######################################
    #  Load image info
    #######################################
    def load_img_info(self):
        img_paths = []
        labels = []

        if self.phase == "train":
            img_paths = os.listdir(os.path.join(self.data_path, "training_selected2"))
        else:
            img_paths = os.listdir(os.path.join(self.data_path, "testing"))
        
        for path in img_paths:
            labels.append(int(path[0]))

        return img_paths, labels

    #######################################
    #  transform
    #######################################
    def get_data_transform(self, phase, rgb_mean, rgb_std):
        transform_info = {
            "rgb_mean": rgb_mean,
            "rgb_std": rgb_std,
        }

        if phase == "train":
            if self.rand_aug:
                self.logger.info(
                    "============= Using Rand Augmentation in Dataset ==========="
                )
                trans = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        RandAugment(),
                        transforms.ToTensor(),
                        transforms.Normalize(rgb_mean, rgb_std),
                    ]
                )
                transform_info["operations"] = [
                    "RandomResizedCrop(64, scale=(0.5, 1.0)),",
                    "RandomHorizontalFlip()",
                    "RandAugment()",
                    "ToTensor()",
                    "Normalize(rgb_mean, rgb_std)",
                ]
            else:
                self.logger.info(
                    "============= Using normal transforms in Dataset ==========="
                )
                trans = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(rgb_mean, rgb_std),
                    ]
                )
                transform_info["operations"] = [
                    "RandomResizedCrop(64, scale=(0.5, 1.0)),",
                    "RandomHorizontalFlip()",
                    "ToTensor()",
                    "Normalize(rgb_mean, rgb_std)",
                ]
        else:
            trans = transforms.Compose(
                [
                    transforms.Resize(81),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize(rgb_mean, rgb_std),
                ]
            )
            transform_info["operations"] = [
                "Resize(81)",
                "CenterCrop(60)",
                "ToTensor()",
                "Normalize(rgb_mean, rgb_std)",
            ]

        return trans


import sys

sys.path.append("../")
from utils.logger_utils import custom_logger

if __name__ == "__main__":
    data = Animal10NLT(
        "train",
        data_path="~/Datasets/animal10n",
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.210, 0.224, 0.225],
        output_path="",
        rand_aug=False,
        logger=custom_logger(output_path="./"),
    )
    indics = []
    for index in indics:
        print(data.img_paths[index])

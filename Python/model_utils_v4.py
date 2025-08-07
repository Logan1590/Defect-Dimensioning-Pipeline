import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import os
import numpy as np
import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T


"""
Main Script Summary
-------------------

This script defines a PyTorch Dataset, data augmentation transforms, and a Mask R-CNN model setup
for training and evaluating instance segmentation on defect images.

Key Components:
1. `DefectSegmentationDataset`:
   - Loads images and their corresponding segmentation masks from a structured folder (with "images" and "masks" subfolders).
   - Handles missing masks by substituting blank masks and printing warnings.
   - Converts masks to instance-level binary masks and bounding boxes compatible with PyTorch detection APIs.

2. `get_transform(train)`:
   - Applies data augmentation when `train=True`, including:
     - Random affine transforms (rotation, translation, shear, scaling)
     - Random perspective distortion
     - Random horizontal and vertical flips
   - Regenerates bounding boxes after mask warping.
   - Always converts PIL images/masks to PyTorch tensors.

3. `get_model_instance_segmentation(num_classes)`:
   - Initializes a pre-trained Mask R-CNN (ResNet-50 FPN backbone).
   - Replaces box and mask heads for the specified number of classes (e.g., 2 for binary: background + defect).

4. `collate_fn`:
   - Custom collation function for batching variable-size image/mask samples in `DataLoader`.

Use Case:
---------
Designed for training a defect segmentation model using PyTorch Lightning or plain PyTorch, particularly for industrial or quality-control datasets with one or more binary masks per image.

Expected Folder Structure:
root_dir/
├── images/
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ... (other input images)
├── masks/
│   ├── sample1.png
│   ├── sample2.png
│   └── ... (corresponding binary masks)

"""


class DefectSegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        img_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")

        image_map = {}
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, img_dir)
                    rel_key = os.path.splitext(rel_path)[0]
                    image_map[rel_key] = full_path

        mask_map = {}
        for root, _, files in os.walk(mask_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, mask_dir)
                    rel_key = os.path.splitext(rel_path)[0]
                    mask_map[rel_key] = full_path

        self.imgs = []
        self.masks = []
        self.mask_missing_flags = []

        for rel_key in sorted(image_map.keys()):
            self.imgs.append(image_map[rel_key])
            if rel_key in mask_map:
                self.masks.append(mask_map[rel_key])
                self.mask_missing_flags.append(False)
            else:
                print(f"⚠️  Warning: Mask for image '{rel_key}' not found. Using blank mask.")
                self.masks.append(None)
                self.mask_missing_flags.append(True)

        print(f"✅ Collected {len(self.imgs)} image entries (some with blank masks if missing).")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        mask_missing = self.mask_missing_flags[idx]

        img = Image.open(img_path).convert("RGB")

        if not mask_missing:
            mask = Image.open(mask_path)
            mask_np = np.array(mask)
        else:
            width, height = img.size
            mask_np = np.zeros((height, width), dtype=np.uint8)

        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[obj_ids != 0]

        masks = mask_np == obj_ids[:, None, None] if len(obj_ids) > 0 else np.zeros((0, *mask_np.shape), dtype=np.uint8)

        boxes = []
        for m in masks:
            pos = np.where(m)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target



def get_transform(train):
    def transform(img, target):
        img = F.to_tensor(img)
        masks = target["masks"].float()

        if train:
            img_pil = F.to_pil_image(img)

            if masks.shape[0] > 0:
                mask_pil_list = [F.to_pil_image(m) for m in masks]

                # Random affine + perspective
                angle = random.uniform(-15, 15)
                translate = (random.uniform(-10, 10), random.uniform(-10, 10))
                scale = random.uniform(0.9, 1.1)
                shear = random.uniform(-10, 10)

                img_pil = F.affine(img_pil, angle=angle, translate=translate, scale=scale, shear=shear)
                mask_pil_list = [F.affine(m, angle=angle, translate=translate, scale=scale, shear=shear)
                                 for m in mask_pil_list]

                if random.random() < 0.5:
                    distortion_scale = 0.3
                    width, height = img_pil.size
                    startpoints, endpoints = T.RandomPerspective.get_params(height, width, distortion_scale=distortion_scale)
                    img_pil = F.perspective(img_pil, startpoints, endpoints)
                    mask_pil_list = [F.perspective(m, startpoints, endpoints) for m in mask_pil_list]

                if random.random() < 0.5:
                    img_pil = F.hflip(img_pil)
                    mask_pil_list = [F.hflip(m) for m in mask_pil_list]

                if random.random() < 0.5:
                    img_pil = F.vflip(img_pil)
                    mask_pil_list = [F.vflip(m) for m in mask_pil_list]

                img = F.to_tensor(img_pil)
                masks = torch.stack([F.to_tensor(m)[0] for m in mask_pil_list])
            else:
                masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)

            # Recalculate boxes if there are masks
            new_boxes = []
            for m in masks:
                pos = (m > 0.5).nonzero(as_tuple=False)
                if pos.numel() == 0:
                    continue
                xmin = pos[:, 1].min().item()
                xmax = pos[:, 1].max().item()
                ymin = pos[:, 0].min().item()
                ymax = pos[:, 0].max().item()
                new_boxes.append([xmin, ymin, xmax, ymax])

            target["boxes"] = torch.as_tensor(new_boxes, dtype=torch.float32) if new_boxes else torch.zeros((0, 4), dtype=torch.float32)
            target["masks"] = masks
            target["labels"] = torch.ones((len(target["boxes"]),), dtype=torch.int64)
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
            target["iscrowd"] = torch.zeros((len(target["boxes"]),), dtype=torch.int64)

        return img, target

    return transform



def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model



def collate_fn(batch):
    return tuple(zip(*batch))

import os
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from model_utils_v4 import (
    DefectSegmentationDataset,
    get_transform,
    get_model_instance_segmentation,
    collate_fn
)



"""
Function Summaries
------------------

train_model(dataset_dir, output_path, split_ratio=0.8, num_epochs=10):
    - Loads a custom segmentation dataset and splits it into training and validation sets.
    - Initializes a Mask R-CNN model for instance segmentation with 2 classes.
    - Trains the model over the specified number of epochs using SGD and a StepLR scheduler.
    - Computes and prints average training and validation loss per epoch.
    - Saves the trained model to disk at the specified output path.
    - Plots training vs. validation loss curves for visual feedback.

"""


def train_model(dataset_dir, output_path, split_ratio=0.8, num_epochs=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load full dataset
    full_dataset = DefectSegmentationDataset(dataset_dir, transforms=get_transform(train=True))

    # Shuffle and split dataset indices
    num_samples = len(full_dataset)
    print('num_samples', num_samples)
    indices = list(range(num_samples))
    random.shuffle(indices)
    split = int(split_ratio * num_samples)
    train_indices = indices[:split]
    val_indices = indices[split:]

    # Train/val datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(
        DefectSegmentationDataset(dataset_dir, transforms=get_transform(train=False)),
        val_indices
    )

    data_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

    model = get_model_instance_segmentation(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_train_loss += losses.item()

            print(f"Epoch {epoch} Iter {i} Train Loss: {losses.item():.4f}")

        avg_train_loss = running_train_loss / len(data_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        # Keep model in training mode to get loss during validation
        model.train()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, targets in data_loader_test:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
                loss_dict = model(images, targets)
                val_loss = sum(loss_dict.values())  # ✅ Now works because loss_dict is actually a dict
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(data_loader_test)
        val_losses.append(avg_val_loss)

        print(f"📘 Epoch {epoch} Summary: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        lr_scheduler.step()

    # Save the model
    torch.save(model.state_dict(), output_path)
    print(f"✅ Training complete. Model saved to {output_path}")

    # --- Plot training & validation loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_losses, label="Train Loss", marker='o')
    plt.plot(range(num_epochs), val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




"""
Main Function Summary
---------------------
- Sets user-defined paths for training data and output model file.
- Configures training parameters (split ratio and number of epochs).
- Calls `train_model()` to train a Mask R-CNN instance segmentation model on the dataset.
- Saves the trained model and displays a loss curve summary.
"""


if __name__ == "__main__":
    
    #=============USER CONFIG================
    # Define filepaths
    DATASET_DIR = r"Path to image folder"
    MODEL_OUT = r"Path to save trained model to"
    SPLIT_RATIO = 0.8  # 80% train / 20% val
    NUM_EPOCHS = 30
    #========================================
    
    
    train_model(DATASET_DIR, MODEL_OUT, SPLIT_RATIO, NUM_EPOCHS)




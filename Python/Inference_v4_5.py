import os
import torch
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks
from model_utils_v4 import get_model_instance_segmentation



"""
Function Summaries
------------------

load_image(path)
    Loads an image from the given path, converts it to RGB, and returns it as a PyTorch tensor.

    Parameters:
        path (str): Path to the image file.

    Returns:
        tensor (torch.Tensor): Image tensor in [C, H, W] format with float values in [0, 1].

load_object_mask(fname, object_masks_dir)
    Loads an object mask and subtracts an optional hole mask if present. Useful for defining a restricted area for evaluation.

    Parameters:
        fname (str): Filename of the target image (with extension).
        object_masks_dir (str): Directory containing object and optional hole masks.

    Returns:
        mask (np.ndarray or None): Boolean array of the object region (with hole excluded if present), or None if mask not found.

run_feature_detection(model_path, images_dir, masks_dir=None, threshold=0.5, visualize=True, object_masks_dir=None)
    Runs instance segmentation using a trained model on all images in the specified directory.

    Parameters:
        model_path (str): Path to the trained model weights (.pth).
        images_dir (str): Directory containing test images (or a single image path).
        masks_dir (str or None): Directory with ground truth binary masks for overlay comparison.
        threshold (float): Confidence threshold for predicted mask activation.
        visualize (bool): Whether to display overlay results via matplotlib.
        object_masks_dir (str or None): Directory containing object masks (and optional holes) for restricted IoU calculation.

    Returns:
        dict: Dictionary mapping each image filename to a tuple:
              (predicted binary mask as numpy array, overlay tensor as [C, H, W] uint8).
"""




def load_image(path):
    
    image = Image.open(path).convert("RGB")
    tensor = F.to_tensor(image)
    return tensor



def load_object_mask(fname, object_masks_dir):

    base_name = os.path.splitext(fname)[0]
    possible_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    object_mask = None
    hole_mask = None

    # Load main object mask
    for ext in possible_exts:
        candidate = os.path.join(object_masks_dir, base_name + ext)
        if os.path.exists(candidate):
            img = Image.open(candidate).convert("L")
            object_mask = np.array(img) > 0
            break

    # Load hole mask if present
    for ext in possible_exts:
        candidate = os.path.join(object_masks_dir, base_name + "_Hole" + ext)
        if os.path.exists(candidate):
            img = Image.open(candidate).convert("L")
            hole_mask = np.array(img) > 0
            break

    if object_mask is None:
        return None  # neither mask found

    if hole_mask is not None:
        return np.logical_and(object_mask, np.logical_not(hole_mask))
    else:
        return object_mask



def run_feature_detection(model_path, images_dir, masks_dir=None, threshold=0.5, visualize=True, object_masks_dir=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = get_model_instance_segmentation(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare image files
    if os.path.isdir(images_dir):
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    else:
        image_files = [os.path.basename(images_dir)]
        images_dir = os.path.dirname(images_dir)

    results = {}

    for fname in image_files:
        fpath = os.path.join(images_dir, fname)
        img_tensor = load_image(fpath).to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        masks = prediction['masks'] > threshold  # tensor shape [N, 1, H, W]
        masks = masks.squeeze(1)  # [N, H, W]

        if masks.size(0) == 0:
            print(f"{fname}: ❌ No features detected")
            combined_mask = torch.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.bool)
            # Use original image for overlay in this case
            overlay_tensor = (img_tensor.cpu() * 255).byte()
        else:
            combined_mask = torch.any(masks, dim=0).cpu()  # combine all masks into one binary mask

            # Convert image tensor to byte format for drawing masks
            vis_tensor = (img_tensor.cpu() * 255).byte()
            overlay_tensor = draw_segmentation_masks(vis_tensor, masks=combined_mask.unsqueeze(0), alpha=0.5, colors="red")

        # If GT masks dir provided, overlay GT masks in green
        if masks_dir:
            base_name = os.path.splitext(fname)[0]
            possible_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            gt_path = None
            for ext in possible_exts:
                candidate = os.path.join(masks_dir, base_name + ext)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break
            if gt_path:
                true_mask_img = Image.open(gt_path).convert("L")
                true_mask = torch.tensor(np.array(true_mask_img) > 0, dtype=torch.bool)
                overlay_tensor = draw_segmentation_masks(overlay_tensor, masks=true_mask.unsqueeze(0), alpha=0.5, colors="green")

        if visualize:
            vis_image = overlay_tensor.permute(1, 2, 0).numpy()
            plt.figure(figsize=(10, 8))
            plt.imshow(vis_image)
            plt.title(f"{fname} - Segmentation Result")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            
        # === If object masks are provided, calculate the intersection over union value within the region defined by the object mask ===  
        if object_masks_dir:
            object_mask = load_object_mask(fname, object_masks_dir)
            if object_mask is None:
                print(f"{fname}: ⚠️ Object mask not found. Skipping IoU calculation.")
            else:
                prediction_mask = combined_mask.numpy()
                restricted_pred = np.logical_and(prediction_mask, object_mask)
                if masks_dir and 'true_mask' in locals():
                    restricted_true = np.logical_and(true_mask.numpy(), object_mask)
                    intersection = np.logical_and(restricted_pred, restricted_true)
                    union = np.logical_or(restricted_pred, restricted_true)
                    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                    print(f"{fname}: Restricted IoU = {iou:.3f}")
                    
                    # Create visualization: green = intersection, red = predicted not in true
                    vis_image = (img_tensor.cpu() * 255).byte().permute(1, 2, 0).numpy()
                    vis_overlay = vis_image.copy()
                    red = [255, 0, 0]
                    green = [0, 255, 0]
                    
                    vis_overlay[np.logical_and(restricted_pred, ~restricted_true)] = red  # False positives
                    vis_overlay[intersection] = green  # True positives
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(vis_overlay)
                    plt.title(f"{fname} - IoU: {iou:.3f}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()


        results[fname] = (combined_mask.numpy(), overlay_tensor)

    print('Defect segmentation successful.')
    return results





"""
Main Function Summary
---------------------
- Loads a trained segmentation model and applies it to each image in the specified folder.
- For each image:
    - Predicts defect masks using the model with a given confidence threshold.
    - Optionally overlays predicted masks (in red) and ground truth masks (in green).
    - If object masks are provided, calculates Intersection-over-Union (IoU) within the object region, and visualizes:
        - True positives (intersection) in green.
        - False positives (predicted-only regions) in red.
- Visualizes and prints the shape of the resulting masks and overlays for each image.
- Returns a dictionary mapping filenames to their predicted masks and overlay tensors.
"""



if __name__ == "__main__":
    
    # ======== USER CONFIG ===================
    model_path = r"Path to trained machine learning model"
    images_dir = r"Path to images folder to apply inference to"
    masks_dir = r"Path to folder holding binary masks of defect segmentation"  # Name the masks the same as the corresponding images
    restriction_masks_dir = r"Path to folder holding binary masks of object segmentation"    # Name the masks the same as the corresponding images
    threshold = 0.4             # Confidence threshold for ML model to classify as defect region
    # =======================

    
    masks_and_overlays = run_feature_detection(
        model_path,
        images_dir,
        masks_dir=masks_dir,
        threshold=threshold,
        visualize=True,
        restriction_masks_dir=restriction_masks_dir  
    )

    for fname, (mask, overlay) in masks_and_overlays.items():
        print(f"{fname}: mask shape = {mask.shape}, overlay tensor shape = {overlay.shape}")

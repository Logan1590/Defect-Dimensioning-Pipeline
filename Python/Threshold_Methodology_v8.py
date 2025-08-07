import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import entropy
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import math

from model_utils_v4 import get_model_instance_segmentation  



"""
Function Summaries
------------------

apply_binary_threshold_mask(one_channel, threshold, above=True):
    - Applies a binary threshold to a single-channel image.
    - Returns a binary mask with pixels set to 255 if they meet the condition (above or below threshold).

kmeans_threshold(img, k=2):
    - Applies K-means clustering (default k=2) on grayscale pixel intensities.
    - Computes a threshold as the midpoint between cluster centers.
    - Returns a binary mask and the applied threshold.

otsu_threshold(img):
    - Applies Otsu’s method to automatically determine the threshold minimizing intra-class variance.
    - Returns a binary mask and the calculated threshold.

quantile_threshold(img, quantile=0.75):
    - Calculates a threshold at the specified quantile of pixel intensities (default: 75%).
    - Returns a binary mask and the applied threshold.

iterative_threshold(img):
    - Uses ISODATA-style iterative averaging to find a stable intensity threshold.
    - Returns a binary mask and the final threshold.

minimum_error_threshold(img):
    - Implements the minimum error thresholding method by minimizing within-class variances in histogram space.
    - Returns a binary mask and the threshold that minimizes classification error.

kapur_entropy_threshold(img):
    - Computes a threshold that maximizes the sum of entropies for foreground and background classes.
    - Returns a binary mask and the best entropy-based threshold.

ml_midpoint_threshold(original_image, model_confidence_threshold=0.05):
    - Applies a pretrained Mask R-CNN model to extract the most confident mask.
    - Computes a threshold as the average of inside vs outside grayscale intensity.
    - Returns a binary mask and the computed threshold.
    - Displays the original image and raw predicted mask.

trained_ml_midpoint_threshold(original_image, model, threshold=0.8):
    - Applies a user-trained Mask R-CNN model to predict the segmentation mask.
    - Computes a threshold from mean inside/outside grayscale intensity.
    - Returns a binary mask and the computed threshold.
    - Displays the original image and predicted mask.

process_image(image_path):
    - Applies all defined thresholding methods to the given image.
    - Displays side-by-side comparison of thresholded masks, along with the original.
    - Returns nothing but provides visual output for analysis.
"""



def apply_binary_threshold_mask(one_channel, threshold, above=True):
    return ((one_channel >= threshold) if above else (one_channel <= threshold)).astype(np.uint8) * 255



def kmeans_threshold(img, k=2):
    pixel_values = img.reshape(-1, 1).astype(np.float32)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixel_values)
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = int((centers[0] + centers[1]) / 2)
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1], threshold



def otsu_threshold(img):
    threshold, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result, int(threshold)



def quantile_threshold(img, quantile=0.75):
    threshold = int(np.quantile(img, quantile))
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1], threshold



def iterative_threshold(img):
    prev_thresh = np.mean(img)
    while True:
        lower = img[img <= prev_thresh]
        upper = img[img > prev_thresh]
        new_thresh = 0.5 * (np.mean(lower) + np.mean(upper))
        if abs(new_thresh - prev_thresh) < 0.5:
            break
        prev_thresh = new_thresh
    return cv2.threshold(img, int(new_thresh), 255, cv2.THRESH_BINARY)[1], int(new_thresh)



def minimum_error_threshold(img):
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / hist.sum()
    cumsum = np.cumsum(hist)
    means = np.cumsum(hist * np.arange(256))
    min_err = np.inf
    best_thresh = 0
    for t in range(1, 255):
        w0, w1 = cumsum[t], 1.0 - cumsum[t]
        if w0 == 0 or w1 == 0: continue
        mu0 = means[t] / w0
        mu1 = (means[-1] - means[t]) / w1
        var0 = np.sum(hist[:t] * (np.arange(t) - mu0) ** 2) / w0
        var1 = np.sum(hist[t:] * (np.arange(t, 256) - mu1) ** 2) / w1
        error = w0 * np.log(var0 + 1e-10) + w1 * np.log(var1 + 1e-10)
        if error < min_err:
            min_err, best_thresh = error, t
    return cv2.threshold(img, best_thresh, 255, cv2.THRESH_BINARY)[1], best_thresh



def kapur_entropy_threshold(img):
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= hist.sum()
    cumsum = np.cumsum(hist)
    max_ent = -np.inf
    best_thresh = 0
    for t in range(1, 255):
        p0, p1 = hist[:t], hist[t:]
        H0 = entropy(p0 / (cumsum[t] + 1e-10), base=2) if cumsum[t] > 0 else 0
        H1 = entropy(p1 / (1 - cumsum[t] + 1e-10), base=2) if (1 - cumsum[t]) > 0 else 0
        total_entropy = H0 + H1
        if total_entropy > max_ent:
            max_ent, best_thresh = total_entropy, t
    return cv2.threshold(img, best_thresh, 255, cv2.THRESH_BINARY)[1], best_thresh



def ml_midpoint_threshold(original_image, model_confidence_threshold=0.05):
    """
    Applies a pretrained Mask R-CNN model to a BGR image.
    Returns a binary mask using mean intensity inside/outside the model mask.
    Plots: original image + raw model segmentation mask.
    """
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    input_tensor = torchvision.transforms.ToTensor()(pil_img).unsqueeze(0)

    with torch.no_grad():
        outputs = _model(input_tensor)

    masks = outputs[0]['masks']
    scores = outputs[0]['scores']
    keep = [i for i, s in enumerate(scores) if s > model_confidence_threshold]

    if not keep:
        print("ML model found no masks above threshold. Falling back to mean.")
        threshold = int(np.mean(gray_image))
        binary_mask = apply_binary_threshold_mask(gray_image, threshold)
        model_mask_display = np.zeros_like(gray_image)
    else:
        idx = max(keep, key=lambda i: torch.count_nonzero(masks[i, 0]))
        raw_mask = masks[idx, 0]
        # Threshold the raw mask and convert to binary display mask
        binary_model_mask = (raw_mask > 0.5).cpu().numpy().astype(np.uint8) * 255
        model_mask_display = binary_model_mask

        # Convert raw float mask to binary
        _, ml_mask = cv2.threshold(model_mask_display, 127, 255, cv2.THRESH_BINARY)

        inside = gray_image[ml_mask == 255].astype(np.float32)
        outside = gray_image[ml_mask == 0].astype(np.float32)

        mean_inside = np.mean(inside) if inside.size > 0 else 0
        mean_outside = np.mean(outside) if outside.size > 0 else 0
        threshold = int((mean_inside + mean_outside) / 2)
        binary_mask = apply_binary_threshold_mask(gray_image, threshold)

    # Plot original + raw model mask
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(model_mask_display, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Pretrained Model Segmentation Mask")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    return binary_mask, threshold



def trained_ml_midpoint_threshold(original_image, model, threshold=0.8):
    """
    Applies a custom Mask R-CNN model to a BGR image.
    Returns a binary mask using mean intensity inside/outside the model mask.
    Plots: original image + raw model segmentation mask.
    """
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    input_tensor = torchvision.transforms.ToTensor()(pil_img).unsqueeze(0)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)[0]

    masks = outputs['masks']
    scores = outputs['scores']
    keep = [i for i, s in enumerate(scores) if s > threshold]

    if not keep:
        print("Custom model: No masks above threshold. Falling back to mean.")
        threshold_value = int(np.mean(gray_image))
        binary_mask = apply_binary_threshold_mask(gray_image, threshold_value)
        model_mask_display = np.zeros_like(gray_image)
    else:
        idx = max(keep, key=lambda i: torch.count_nonzero(masks[i, 0]))
        raw_mask = masks[idx, 0]
        # Threshold the raw mask and convert to binary display mask
        binary_model_mask = (raw_mask > 0.5).cpu().numpy().astype(np.uint8) * 255
        model_mask_display = binary_model_mask

        _, ml_mask = cv2.threshold(model_mask_display, 127, 255, cv2.THRESH_BINARY)

        inside = gray_image[ml_mask == 255].astype(np.float32)
        outside = gray_image[ml_mask == 0].astype(np.float32)

        mean_inside = np.mean(inside) if inside.size > 0 else 0
        mean_outside = np.mean(outside) if outside.size > 0 else 0
        threshold_value = int((mean_inside + mean_outside) / 2)
        binary_mask = apply_binary_threshold_mask(gray_image, threshold_value)

    # Plot original + raw model mask
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(model_mask_display, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Retrained Model Segmentation Mask")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    return binary_mask, threshold_value



# === Main Processing Function ===
def process_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = {"Original": (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "N/A")}
    for name, func in methods.items():
        raw_mask, threshold = func(image)
        results[name] = (raw_mask, threshold)


    # === Plot Thresholding Results ===
    num_methods = len(results)
    cols = 3
    rows = math.ceil(num_methods / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(f"Binary Thresholding on '{image_path.name}'", fontsize=14)
    
    # Flatten axs for easy indexing (even if 1 row)
    axs = axs.flat if isinstance(axs, np.ndarray) else [axs]
    
    all_keys = list(results.keys())
    for i, key in enumerate(all_keys):
        ax = axs[i]
        img, threshold = results[key]
        cmap = None if key == "Original" else 'gray'
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255 if cmap == 'gray' else None)
        title = key if threshold == "N/A" else f"{key}\nThreshold: {threshold}"
        ax.set_title(title)
        ax.axis('off')

    # Turn off unused subplots
    for j in range(len(all_keys), len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()






"""
Main Function Summary
---------------------
- Defines user paths and thresholds for classical and ML-based segmentation.
- Loads pretrained and custom Mask R-CNN models.
- Defines a dictionary of thresholding methods including ML-based and statistical approaches.
- If a single image is provided, processes it with all thresholding methods.
- If a directory is provided, iterates over all valid image files and processes each.
"""

if __name__ == "__main__":


    # ===========USER CONFIG===================
    # Define filepaths
    path = r'Path to image folder'
    custom_model_path = r"Path to trained object segmentation ML model"
    
    # Define method settings
    trained_model_confidence_threshold=0.05
    model_confidence_threshold=0.05
    quantile_setting = 0.8
    # ===========================================
    
    
    # === Load Model Once ===
    _weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    _model = maskrcnn_resnet50_fpn_v2(weights=_weights)
    _model.eval()
    
    
    # === Load Custom Model Once ===
    custom_model = get_model_instance_segmentation(num_classes=2)
    custom_model.load_state_dict(torch.load(custom_model_path, map_location='cpu'))  # or 'cuda' if needed
    custom_model.eval()
    
    
    # === Thresholding Methods ===
    methods = {
        "K-Means": lambda img: kmeans_threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        "Otsu's": lambda img: otsu_threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        f"Quantile ({quantile_setting:.2f})": lambda img: quantile_threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), quantile_setting),
        "Iterative (ISODATA)": lambda img: iterative_threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        "Minimum Error": lambda img: minimum_error_threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        "Kapur's Entropy": lambda img: kapur_entropy_threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        "ML Midpoint": ml_midpoint_threshold,
        "Trained ML Midpoint": lambda img: trained_ml_midpoint_threshold(img, custom_model, trained_model_confidence_threshold)
    }
    
    
    
    # === Process Images (One or more) ===
    p = Path(path)
    if p.is_file():
        process_image(p)
    elif p.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        images = sorted([f for f in p.iterdir() if f.suffix.lower() in image_extensions])
        if not images:
            print(f"No images found in: {p}")
        for img_file in images:
            process_image(img_file)
    else:
        print(f"Invalid path: {path}")

import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import os
import glob
from scipy.spatial.distance import cdist
from scipy.stats import entropy


"""
Function Summaries
------------------

load_image(image_path)
    Loads an image from a given path, converts it to RGB and grayscale formats, 
    and returns both as NumPy arrays along with a PyTorch tensor for model inference.

get_ml_mask(input_tensor, model, confidence_threshold=0.05)
    Generates a binary segmentation mask using a pre-trained Mask R-CNN model, 
    selecting the largest detected region above the confidence threshold.

iterative_threshold(img)
    Computes a global threshold using the iterative ISODATA algorithm based on the mean intensity of pixel groups.

minimum_error_threshold(img)
    Calculates an optimal threshold by minimizing within-class variance, based on Kittler and Illingworth’s minimum error approach.

kapur_entropy_threshold(img)
    Determines a threshold by maximizing the sum of entropies of two pixel classes (foreground/background).

ml_mean_threshold(gray, ml_mask)
    Computes a threshold as the mean of average intensities inside and outside a machine-learned mask.

clean_largest_region(mask)
    Cleans a binary mask by keeping only the largest connected component.

rank_final_corners(all_corners_with_methods, max_corners, close_dist=7, sparsity_radius=300, beta=0.01)
    Ranks candidate corner points based on method diversity and spatial sparsity. 
    Selects the top corners while enforcing a minimum spacing constraint.

plot_in_method_scores(image_np, method_to_corners, method_to_scores)
    Plots detected corners per thresholding method with their respective in-method scores.

detect_corners_on_edges(mask, max_corners=30, use_approx=True, approx_epsilon_frac=0.001,
                        sigma=30, min_edge_length=20, isolation_distance=50, isolation_boost=3, density_radius=200)
    Detects corner points from polygonal approximations of mask contours. 
    Scores corners based on local angle (closeness to 90°) and spatial isolation.

plot_thresholds_and_corners(gray, all_thresholds_with_corners)
    Visualizes thresholded images for each method, overlaying detected corner points.

plot_all_corners_with_scores(image_np, all_corners_with_methods, final_scores_for_points)
    Plots all candidate corners from all methods along with their computed final scores.

run_detection_pipeline(img_path, max_corners=30, visualize=True)
    Performs the full detection pipeline:
        1. Loads the image and runs Mask R-CNN to obtain the object mask.
        2. Applies multiple thresholding methods.
        3. Extracts candidate corners and scores them.
        4. Ranks and filters corners to select the top features.
        5. Optionally visualizes intermediate and final results.
    Returns grayscale image, ML mask, and final corner coordinates.

"""



# === Image + Mask Loading ===
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    return image_np, gray, tensor



def get_ml_mask(input_tensor, model, confidence_threshold=0.05):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    masks = outputs[0]['masks']
    scores = outputs[0]['scores']
    keep_idxs = [i for i, s in enumerate(scores) if s > confidence_threshold]
    if not keep_idxs:
        print("No high-confidence detections found. Returning blank mask.")
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.uint8)
    largest_mask_idx = max(keep_idxs, key=lambda i: torch.count_nonzero(masks[i, 0]))
    mask = masks[largest_mask_idx, 0].mul(255).byte().cpu().numpy()
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask



# === Thresholding Methods ===
def iterative_threshold(img):
    prev_thresh = np.mean(img)
    while True:
        lower = img[img <= prev_thresh]
        upper = img[img > prev_thresh]
        new_thresh = 0.5 * (np.mean(lower) + np.mean(upper))
        if abs(new_thresh - prev_thresh) < 0.5:
            break
        prev_thresh = new_thresh
    return int(new_thresh)



def minimum_error_threshold(img):
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / hist.sum()
    cumsum = np.cumsum(hist)
    means = np.cumsum(hist * np.arange(256))
    min_err = np.inf
    best_thresh = 0
    for t in range(1, 255):
        w0, w1 = cumsum[t], 1.0 - cumsum[t]
        if w0 == 0 or w1 == 0:
            continue
        mu0 = means[t] / w0
        mu1 = (means[-1] - means[t]) / w1
        var0 = np.sum(hist[:t] * (np.arange(t) - mu0) ** 2) / w0
        var1 = np.sum(hist[t:] * (np.arange(t, 256) - mu1) ** 2) / w1
        error = w0 * np.log(var0 + 1e-10) + w1 * np.log(var1 + 1e-10)
        if error < min_err:
            min_err, best_thresh = error, t
    return best_thresh



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
    return best_thresh



def ml_mean_threshold(gray, ml_mask):
    if np.count_nonzero(ml_mask) == 0:
        return int(np.mean(gray))
    inside_vals = gray[ml_mask == 255].astype(np.float32)
    outside_vals = gray[ml_mask == 0].astype(np.float32)
    mean_inside = np.mean(inside_vals) if inside_vals.size > 0 else 0
    mean_outside = np.mean(outside_vals) if outside_vals.size > 0 else 0
    return int((mean_inside + mean_outside) / 2)



def clean_largest_region(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels == largest_label] = 255
    return cleaned_mask



def rank_final_corners(all_corners_with_methods, max_corners, close_dist=7,
                       sparsity_radius=300, beta=0.01):
    if not all_corners_with_methods:
        return [], {}

    pts = np.array([pt for pt, _ in all_corners_with_methods])
    methods = [m for _, m in all_corners_with_methods]

    # Distance matrix (between all corners)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)

    # Method diversity boost: more methods near a point within close_dist = higher score
    method_boosts = []
    for i, pt in enumerate(pts):
        close_idxs = np.where(dists[i] <= close_dist)[0]
        unique_methods = set([methods[i]] + [methods[j] for j in close_idxs])
        boost = len(unique_methods) ** 2
        method_boosts.append(boost)
    method_boosts = np.array(method_boosts)

    # Sparsity penalty: consider neighbors at distance > close_dist and <= sparsity_radius
    sparsity_penalties = np.zeros(len(pts))
    for i in range(len(pts)):
        mask = (dists[i] > close_dist) & (dists[i] <= sparsity_radius)
        relevant_dists = dists[i][mask]
        if relevant_dists.size > 0:
            decay_vals = np.exp(-beta * (relevant_dists - close_dist) ** 2)
            sparsity_penalties[i] = decay_vals.sum()
        else:
            sparsity_penalties[i] = 0.0

    sparsity_rewards = 1 / (1 + sparsity_penalties)

    # Final score = method boost * sparsity reward
    final_scores = method_boosts * sparsity_rewards

    final_scores_for_points = {tuple(pt): float(score) for pt, score in zip(pts, final_scores)}

    # Filter top max_corners with spacing enforcement
    sorted_indices = np.argsort(-final_scores)
    filtered_corners = []
    seen = []
    for idx in sorted_indices:
        pt = tuple(pts[idx])
        if all(np.linalg.norm(np.array(pt) - np.array(s)) > close_dist for s in seen):
            filtered_corners.append(pt)
            seen.append(pt)
            if len(filtered_corners) >= max_corners:
                break

    return filtered_corners, final_scores_for_points



def plot_in_method_scores(image_np, method_to_corners, method_to_scores):
    """
    Plots in-method corner scores for each thresholding method.

    Args:
        image_np: Grayscale or RGB image.
        method_to_corners: Dict of method name to list of corner (x, y) tuples.
        method_to_scores: Dict of method name to list of scores (same order as corners).
    """
    import matplotlib.pyplot as plt

    n = len(method_to_corners)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))

    if n == 1:
        axes = [axes]

    for ax, (method, corners) in zip(axes, method_to_corners.items()):
        scores = method_to_scores.get(method, [])
        ax.imshow(image_np)
        ax.set_title(f"{method} — In-Method Scores (n={len(corners)})")
        ax.axis('off')

        for (x, y), score in zip(corners, scores):
            ax.plot(x, y, 'o', color='lime', markersize=6, markeredgecolor='black', markeredgewidth=1)
            ax.text(x + 5, y - 5, f"{score:.0f}", color='yellow', fontsize=6, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

    plt.tight_layout()
    plt.show()



def detect_corners_on_edges(mask, max_corners=30, use_approx=True, approx_epsilon_frac=0.001,
                            sigma=30, min_edge_length=20, isolation_distance=50, isolation_boost=3, density_radius=200):
    def enforce_min_edge_length(polygon, min_length):
        polygon = polygon.reshape(-1, 2).tolist()
        changed = True
        while changed:
            changed = False
            new_polygon = []
            i = 0
            n = len(polygon)
            while i < n:
                pt1 = polygon[i]
                pt2 = polygon[(i + 1) % n]
                edge_vec = np.array(pt2) - np.array(pt1)
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < min_length:
                    midpoint = ((np.array(pt1) + np.array(pt2)) / 2).astype(int).tolist()
                    new_polygon.append(midpoint)
                    i += 2
                    changed = True
                else:
                    new_polygon.append(pt1)
                    i += 1
            polygon = new_polygon
            if len(polygon) < 3:
                break
        return np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)

    def angle_between(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if use_approx:
        approx_contours = []
        for cnt in contours:
            epsilon = approx_epsilon_frac * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx = enforce_min_edge_length(approx, min_length=min_edge_length)
            approx_contours.append(approx)
        contours = approx_contours

    polygon_candidates = []
    angle_dict = {}
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        if len(cnt) < 3:
            continue
        for i in range(len(cnt)):
            p1 = cnt[i - 1]
            p2 = cnt[i]
            p3 = cnt[(i + 1) % len(cnt)]
            angle = angle_between(p1, p2, p3)
            angle_penalty = np.exp(-((angle - 90) ** 2) / (2 * sigma ** 2)) if angle < 90 else 1.0
            polygon_candidates.append((tuple(p2), angle))
            angle_dict[tuple(p2)] = angle

    sorted_polygon_corners = []
    if polygon_candidates:
        pts = np.array([pt for pt, _ in polygon_candidates])
        angles = np.array([angle for _, angle in polygon_candidates])
        scores = np.exp(-((angles - 90) ** 2) / (2 * sigma ** 2))

        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        local_density = np.sum((dists < density_radius).astype(np.float32), axis=1)
        final_scores = scores / (1 + local_density)
        for i in range(len(pts)):
            if np.min(dists[i]) > isolation_distance:
                final_scores[i] *= isolation_boost

        sorted_indices = np.argsort(-final_scores)
        seen = []
        for idx in sorted_indices:
            pt = tuple(pts[idx])
            if all(np.linalg.norm(np.array(pt) - np.array(s)) > 2 for s in seen):
                sorted_polygon_corners.append(pt)
                seen.append(pt)
                if len(sorted_polygon_corners) >= max_corners:
                    break

    edges_contour = np.zeros_like(mask_bin)
    cv2.drawContours(edges_contour, contours, -1, color=255, thickness=20)
    result_img = cv2.cvtColor(mask_bin.copy(), cv2.COLOR_GRAY2RGB)
    result_img[edges_contour > 0] = (128, 255, 255)
    for pt in sorted_polygon_corners:
        cv2.circle(result_img, pt, 5, (0, 255, 0), -1)

    return result_img, sorted_polygon_corners, angle_dict



def plot_thresholds_and_corners(gray, all_thresholds_with_corners):
    n = len(all_thresholds_with_corners)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 6))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title("Original Grayscale")
    axes[0].axis('off')
    
    for i, (method, data) in enumerate(all_thresholds_with_corners.items()):
        color_img = data["visual"]
        corners = data["corners"]
        
        axes[i + 1].imshow(color_img)
        axes[i + 1].set_title(f"{method}\nT={data['threshold']}")
        axes[i + 1].axis('off')
        
        # Plot corners as green dots
        if corners:
            corners_arr = np.array(corners)
            axes[i + 1].scatter(corners_arr[:, 0], corners_arr[:, 1], c='lime', s=10, edgecolors='black', linewidths=0.5)

    plt.tight_layout()
    plt.show()
    
    
    
def plot_all_corners_with_scores(image_np, all_corners_with_methods, final_scores_for_points):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_np)
    ax.set_title(f"All Candidate Corners with Scores (n={len(all_corners_with_methods)})")
    ax.axis('off')

    for pt, method in all_corners_with_methods:
        score = final_scores_for_points.get(pt, None)
        if score is None:
            continue
        x, y = pt
        ax.plot(x, y, 'o', color='lime', markersize=6, markeredgecolor='black', markeredgewidth=1)
        ax.text(x + 5, y - 5, f"{score:.0f}", color='yellow', fontsize=6, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

    plt.tight_layout()
    plt.show()



def run_detection_pipeline(img_path, max_corners=30, visualize=True):
    import matplotlib.pyplot as plt

    # === Load Image and Model ===
    image_np, gray, input_tensor = load_image(img_path)
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    ml_mask = get_ml_mask(input_tensor, model)

    # === Compute Thresholds ===
    thresholds = {
        "ML Mean Threshold": ml_mean_threshold(gray, ml_mask),
        "Iterative ISODATA": iterative_threshold(gray),
        "Minimum Error": minimum_error_threshold(gray),
        "Kapur Entropy": kapur_entropy_threshold(gray)
    }

    # === Use all thresholds without filtering ===
    unique_thresholds = thresholds

    all_corners_with_methods = {}
    visuals_per_method = {}
    corner_angles = {}

    for method_name, thresh_val in unique_thresholds.items():
        mask = (gray > thresh_val).astype(np.uint8) * 255
        cleaned = clean_largest_region(mask)
        visual, corners, angle_dict = detect_corners_on_edges(cleaned, max_corners=max_corners)
        all_corners_with_methods[method_name] = corners
        visuals_per_method[method_name] = {
            "mask": cleaned,
            "corners": corners,
            "threshold": thresh_val,
            "visual": visual
        }
        for pt in corners:
            corner_angles[pt] = angle_dict.get(pt, 90)

    # === Collect in-method scores for plotting
    method_to_inmethod_scores = {}
    for method, data in visuals_per_method.items():
        corners = data["corners"]
        mask = data["mask"]
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
        if not corners:
            method_to_inmethod_scores[method] = []
            continue
    
        pts = np.array(corners)
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        angles = [corner_angles.get(pt, 90) for pt in corners]
        angle_scores = np.exp(-((np.array(angles) - 90) ** 2) / (2 * 30 ** 2))  # sigma=30
        local_density = np.sum((dists < 200).astype(np.float32), axis=1)  # density_radius=200
        final_scores = angle_scores / (1 + local_density)
        for i, pt in enumerate(pts):
            if np.min(dists[i]) > 50:  # isolation_distance=50
                final_scores[i] *= 3   # isolation_boost=3
        method_to_inmethod_scores[method] = final_scores


    # === Plot per-method visuals ===
    if visualize:
        plot_thresholds_and_corners(gray, visuals_per_method)
        
        plot_in_method_scores(image_np, {k: v["corners"] for k, v in visuals_per_method.items()}, method_to_inmethod_scores)


    # === Combine all corners from all methods ===
    combined = []
    for method, pts in all_corners_with_methods.items():
        combined.extend([(pt, method) for pt in pts])

    # === Rank and filter final corners using updated scoring ===
    filtered_corners, final_scores_for_points = rank_final_corners(
        combined,
        max_corners=max_corners,
        close_dist=7,
        sparsity_radius=100,
        beta=0.01
    )


    if visualize:
        # === Plot all candidate corners with scores ===
        plot_all_corners_with_scores(image_np, combined, final_scores_for_points)

        # === Final overlay with filtered corners and scores ===
        final_overlay = image_np.copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(final_overlay)
        ax.set_title(f"{os.path.basename(img_path)} — Final Corners (n={len(filtered_corners)})")
        ax.axis('off')
    
        for pt in filtered_corners:
            x, y = pt
            score = final_scores_for_points.get(pt, 0)
            ax.plot(x, y, 'o', color='lime', markersize=6, markeredgecolor='black', markeredgewidth=1)
            ax.text(x + 5, y - 5, f"{score:.0f}", color='yellow', fontsize=6, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))
    
        plt.tight_layout()
        plt.show()
    
    print("Feature points extracted.")

    return {
        "gray": gray,
        "ml_mask": ml_mask,
        "corner_coords": filtered_corners
    }





"""

Main Function Summary
---------------------

This script performs multi-method corner detection on input images using a combination of 
deep-learning-based segmentation and classical thresholding methods. The pipeline includes:
1. Running Mask R-CNN to obtain a primary object mask.
2. Applying multiple thresholding techniques (ISODATA, minimum error, entropy, ML-based).
3. Detecting contour-based corners and scoring them for method diversity and isolation.
4. Selecting top corner points based on a combined ranking metric.
5. Optionally visualizing thresholded images, corner scores, and final detections.
The script can process a single image or a directory of images.
"""


# === Entry Point ===
if __name__ == "__main__":

    #============USER CONFIG============
    # Define filepath
    input_path = r"Path to new image"
    #=====================================


    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    if os.path.isfile(input_path) and input_path.lower().endswith(supported_exts):
        run_detection_pipeline(input_path, max_corners=20)
    elif os.path.isdir(input_path):
        image_files = [f for ext in supported_exts for f in glob.glob(os.path.join(input_path, f'*{ext}'))]
        image_files.sort()
        for img_path in image_files:
            print(f"\n=== Processing: {os.path.basename(img_path)} ===")
            run_detection_pipeline(img_path, max_corners=20)
    else:
        print("Invalid path. Provide a valid image or folder.")


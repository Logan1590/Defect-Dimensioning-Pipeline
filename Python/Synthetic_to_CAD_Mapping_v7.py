import cv2
import numpy as np
import matplotlib.pyplot as plt

from ML_Segmentation_v25 import run_detection_pipeline



"""
Function Summaries
------------------

filter_matches_by_best_proximity(good_matches, kp1, kp2, max_pixel_dist=None):
    - Filters SIFT matches by spatial proximity.
    - Keeps only matches that are the closest for both the reference and real keypoints.
    - Optionally removes matches beyond a pixel distance threshold.

sift_match_and_solvePnP(real_img_path, reference_img_path, K, dist_coeffs=None, max_keypoints=50):
    - Uses a custom corner detector to extract keypoints in both real and reference images.
    - Computes SIFT descriptors and matches them using Lowe’s ratio test.
    - Applies additional spatial filtering using reciprocal nearest match distance.
    - Visualizes matched corner keypoints with unique indices.
    - Returns both the list of filtered `cv2.DMatch` objects and index pairs (reference_idx → real_idx).
"""



def filter_matches_by_best_proximity(good_matches, kp1, kp2, max_pixel_dist=None):
    """
    Filters SIFT matches by:
    - Keeping only the closest (by pixel distance) match per keypoint (from both images).
    - Optionally applying a global pixel distance threshold.

    Parameters:
    - good_matches: list of cv2.DMatch from knnMatch and Lowe's ratio test
    - kp1: keypoints in reference image (queryIdx)
    - kp2: keypoints in real image (trainIdx)
    - max_pixel_dist: float or None. If set, discard matches beyond this pixel distance.

    Returns:
    - List of filtered cv2.DMatch objects
    """
    best_matches_by_synth = {}  # queryIdx -> (match, distance)
    best_matches_by_real = {}   # trainIdx -> (match, distance)

    for m in good_matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))

        # Optional distance threshold filter
        if max_pixel_dist is not None and dist > max_pixel_dist:
            print(f"[Filter Debug] Match ({m.queryIdx}->{m.trainIdx}) rejected due to pixel distance threshold: {dist:.2f} > {max_pixel_dist}")
            continue

        # Keep best match per reference keypoint
        prev_synth = best_matches_by_synth.get(m.queryIdx)
        if prev_synth is None or dist < prev_synth[1]:
            best_matches_by_synth[m.queryIdx] = (m, dist)

        # Keep best match per real keypoint
        prev_real = best_matches_by_real.get(m.trainIdx)
        if prev_real is None or dist < prev_real[1]:
            best_matches_by_real[m.trainIdx] = (m, dist)

    # Only retain matches that are best in both directions
    final_matches = []
    for m, dist in best_matches_by_synth.values():
        if best_matches_by_real[m.trainIdx][0] == m:
            final_matches.append(m)
        else:
            print(f"[Filter Debug] Match ({m.queryIdx}->{m.trainIdx}) rejected: not best reciprocal match")

    return final_matches



def sift_match_and_solvePnP(real_img_path, reference_img_path, K, dist_coeffs=None, max_keypoints=50):
    real_img = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)
    reference_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)

    real_coords = run_detection_pipeline(real_img_path, max_corners=20)["corner_coords"]
    synth_coords = run_detection_pipeline(reference_img_path, max_corners=20)["corner_coords"]

    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    kp_real = [cv2.KeyPoint(float(x), float(y), 15) for (x, y) in real_coords]
    kp_synth = [cv2.KeyPoint(float(x), float(y), 15) for (x, y) in synth_coords]

    _, des_real = sift.compute(real_img, kp_real)
    _, des_synth = sift.compute(reference_img, kp_synth)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_synth, des_real, k=2)

    good_matches = []
    ratio = 0.95
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    # Before spatial filtering
    print(f"\nGood matches before spatial filtering: {len(good_matches)}")
    for m in good_matches:
        synth_idx = m.queryIdx
        real_idx = m.trainIdx
    
        # Extract coordinates from keypoints
        synth_pt = kp_synth[synth_idx].pt  # (x, y)
        real_pt = kp_real[real_idx].pt     # (x, y)
        euclidean_dist = np.linalg.norm(np.array(synth_pt) - np.array(real_pt))
    
        print(
            f"Good Match: synth_idx={synth_idx}, real_idx={real_idx}, "
            f"descriptor_distance={m.distance:.2f}, "
            f"euclidean_distance={euclidean_dist:.2f}, "
            f"synth_pt=({synth_pt[0]:.1f}, {synth_pt[1]:.1f}), "
            f"real_pt=({real_pt[0]:.1f}, {real_pt[1]:.1f})"
        )

    # --- New: Filter by spatial proximity ---
    good_matches = filter_matches_by_best_proximity(good_matches, kp_synth, kp_real, max_pixel_dist=820)
    # ---------------------------------------

    print(f"\nGood matches after spatial filtering: {len(good_matches)}")
    for m in good_matches:
        synth_idx = m.queryIdx
        real_idx = m.trainIdx
    
        # Extract coordinates from keypoints
        synth_pt = kp_synth[synth_idx].pt  # (x, y)
        real_pt = kp_real[real_idx].pt     # (x, y)
        euclidean_dist = np.linalg.norm(np.array(synth_pt) - np.array(real_pt))
    
        print(
            f"Filtered Match: synth_idx={synth_idx}, real_idx={real_idx}, "
            f"descriptor_distance={m.distance:.2f}, "
            f"euclidean_distance={euclidean_dist:.2f}, "
            f"synth_pt=({synth_pt[0]:.1f}, {synth_pt[1]:.1f}), "
            f"real_pt=({real_pt[0]:.1f}, {real_pt[1]:.1f})"
        )

    matched_synth_idxs = {m.queryIdx for m in good_matches}
    matched_real_idxs = {m.trainIdx for m in good_matches}


    def draw_indexed_keypoints(image, coords, matched_idx_set):
        img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        for idx, (x, y) in enumerate(coords):
            color = (0, 255, 0) if idx in matched_idx_set else (255, 0, 0)
            cv2.circle(img_rgb, (int(x), int(y)), 12, color, -1)
            cv2.putText(img_rgb, str(idx), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)
        return img_rgb

    real_annotated = draw_indexed_keypoints(real_img, real_coords, matched_real_idxs)
    synth_annotated = draw_indexed_keypoints(reference_img, synth_coords, matched_synth_idxs)

    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    axs[0].imshow(cv2.cvtColor(synth_annotated, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Reference Image (Indexed Corner Points)")
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(real_annotated, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Real Image (Indexed Corner Points)")
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

    matched_indices = [(m.queryIdx, m.trainIdx) for m in good_matches]
    return good_matches, matched_indices




"""
Main Function Summary
---------------------
- Configures the paths and camera intrinsics.
- Runs the full SIFT-based corner matching pipeline.
- Prints out the matched index pairs between reference and real images.
- Includes visualization of matched keypoints with index annotations for debugging and inspection.
"""

if __name__ == "__main__":

    #==========USER CONFIG===========
    # Define filepaths
    real_img_path = r"Path to new image"
    reference_img_path = r"Path to reference image"
     
    # Define camera intrinsics
    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    #=================================
    
    
    
    try:
        matches, idx_pairs = sift_match_and_solvePnP(real_img_path, reference_img_path, K, dist_coeffs)
        print("Matched pairs (reference_idx -> real_idx):", idx_pairs)
    
    except Exception as e:
        print("Error:", e)


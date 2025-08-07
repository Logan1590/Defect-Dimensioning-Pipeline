import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from ML_Segmentation_v26 import run_detection_pipeline 
from Eye_in_Hand_Calibration_v4 import retrieve_eyehand_calibration



"""
Function Summaries
------------------

load_reference_keypoints(json_path, image_name)
    Loads 2D keypoints from a JSON file for a given image and associates them with 3D CAD points using ID matching. 
    Returns image and object point lists.

filter_matches_by_best_proximity(good_matches, kp1, kp2, max_pixel_dist=None)
    Filters SIFT keypoint matches by selecting only reciprocal closest matches within an optional pixel distance threshold. 
    Improves robustness of 2D–3D correspondences.

plot_keypoints_and_matches(real_img, reference_img, kp_real, kp_reference, good_matches)
    Visualizes matched and unmatched keypoints in both real and reference images, showing index labels and coloring matched points in green.

rvec_tvec_to_pose6d(rvec, tvec)
    Converts a rotation vector (rvec) and translation vector (tvec) into a 6DOF pose [x, y, z, roll, pitch, yaw], with angles in degrees.

remove_worst_pixel_match(good_matches, kp_reference, kp_real)
    Removes the match with the largest 2D pixel distance between reference and real keypoints.
    Used iteratively to improve pose estimation when solvePnPRansac fails.

remove_pixel_outliers_by_zscore(good_matches, kp_reference, kp_real, z_thresh=2.5)
    Detects and removes matches with outlier pixel distances based on z-score analysis.
    Enhances pose estimation stability by removing spatially inconsistent matches.

estimate_pose(real_img_path, reference_img_path, json_path, K, dist_coeffs=None, max_keypoints=50, compute_ref_pose=False, visualize=False)
    Estimates the 6DOF object pose from a real image using solvePnPRansac and a feature-matching pipeline based on SIFT keypoints and 2D–3D correspondences.
    Optionally computes and refines the pose for the reference image as well.
    Applies filtering and fallback reattempts with progressive match culling.
    Returns rvec and tvec for real and reference images, along with matched keypoint coordinates.

"""





def load_reference_keypoints(json_path, image_name):
    with open(json_path, "r") as f:
        data = json.load(f)

    cad_lookup = data.get("cad_points")
    if cad_lookup is None:
        raise ValueError("Missing 'cad_points' key in JSON.")

    image_data = data.get(image_name)
    if image_data is None:
        raise ValueError(f"No keypoints found for image: {image_name}")

    img_points = []
    obj_points = []

    for kp in image_data:
        point_id = str(kp["ID"])  # Ensure string key for JSON dict
        if point_id not in cad_lookup:
            raise KeyError(f"CAD point ID '{point_id}' not found in 'cad_points'.")
        img_points.append(tuple(kp["image_point"]))
        obj_points.append(tuple(cad_lookup[point_id]))

    return img_points, obj_points



def filter_matches_by_best_proximity(good_matches, kp1, kp2, max_pixel_dist=None):
    best_matches_by_reference = {}
    best_matches_by_real = {}
    for m in good_matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if max_pixel_dist is not None and dist > max_pixel_dist:
            print(f"[Filter Debug] Match ({m.queryIdx}->{m.trainIdx}) rejected due to pixel distance threshold: {dist:.2f} > {max_pixel_dist}")
            continue
        if m.queryIdx not in best_matches_by_reference or dist < best_matches_by_reference[m.queryIdx][1]:
            best_matches_by_reference[m.queryIdx] = (m, dist)
        if m.trainIdx not in best_matches_by_real or dist < best_matches_by_real[m.trainIdx][1]:
            best_matches_by_real[m.trainIdx] = (m, dist)
    final_matches = []
    for m, dist in best_matches_by_reference.values():
        if best_matches_by_real[m.trainIdx][0] == m:
            final_matches.append(m)
        else:
            print(f"[Filter Debug] Match ({m.queryIdx}->{m.trainIdx}) rejected: not best reciprocal match")
    return final_matches



def plot_keypoints_and_matches(real_img, reference_img, kp_real, kp_reference, good_matches):

    matched_ref_idxs = {m.queryIdx for m in good_matches}
    matched_real_idxs = {m.trainIdx for m in good_matches}

    real_coords = [kp.pt for kp in kp_real]
    ref_coords = [kp.pt for kp in kp_reference]

    def draw_indexed_keypoints(image, coords, matched_idx_set):
        img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        for idx, (x, y) in enumerate(coords):
            color = (0, 255, 0) if idx in matched_idx_set else (255, 0, 0)
            cv2.circle(img_rgb, (int(x), int(y)), 12, color, -1)
            cv2.putText(img_rgb, str(idx), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)
        return img_rgb

    real_annotated = draw_indexed_keypoints(real_img, real_coords, matched_real_idxs)
    ref_annotated = draw_indexed_keypoints(reference_img, ref_coords, matched_ref_idxs)

    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    axs[0].imshow(cv2.cvtColor(ref_annotated, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Reference Image (Indexed Corner Points)")
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(real_annotated, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Real Image (Indexed Corner Points)")
    axs[1].axis('off')
    plt.tight_layout()
    plt.show(block=True)

    

def rvec_tvec_to_pose6d(rvec, tvec):
    """
    Converts OpenCV rvec and tvec to a 6-DOF pose [x, y, z, roll, pitch, yaw],
    with angles in degrees.
    """
    x, y, z = tvec.ravel()
    R_mat, _ = cv2.Rodrigues(rvec)
    euler = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
    roll, pitch, yaw = euler
    return [x, y, z, roll, pitch, yaw]



def remove_worst_pixel_match(good_matches, kp_reference, kp_real):
    """
    Removes the match with the largest 2D pixel distance between reference and real keypoints.

    Args:
        good_matches: List of cv2.DMatch
        kp_reference: List of cv2.KeyPoint (reference image)
        kp_real: List of cv2.KeyPoint (real image)

    Returns:
        List of cv2.DMatch with the worst match removed
    """
    if len(good_matches) <= 4:
        # Don't remove if already at or below minimum for PnP
        return good_matches

    # Compute pixel distances
    distances = []
    for m in good_matches:
        pt_ref = np.array(kp_reference[m.queryIdx].pt)
        pt_real = np.array(kp_real[m.trainIdx].pt)
        dist = np.linalg.norm(pt_ref - pt_real)
        distances.append(dist)

    # Index of the worst match
    worst_idx = np.argmax(distances)
    print(f"[Culling] Removing match with max pixel distance: {distances[worst_idx]:.2f} px")

    # Return new list without the worst match
    new_matches = good_matches[:worst_idx] + good_matches[worst_idx+1:]
    return new_matches



def remove_pixel_outliers_by_zscore(good_matches, kp_reference, kp_real, z_thresh=2.5):
    distances = []
    for m in good_matches:
        pt_ref = np.array(kp_reference[m.queryIdx].pt)
        pt_real = np.array(kp_real[m.trainIdx].pt)
        dist = np.linalg.norm(pt_ref - pt_real)
        distances.append(dist)

    distances = np.array(distances)
    median = np.median(distances)
    std = np.std(distances)

    filtered_matches = []
    for i, m in enumerate(good_matches):
        if std > 0 and (distances[i] - median) / std > z_thresh:
            print(f"[Outlier Culling] Match {m.queryIdx}->{m.trainIdx} removed (distance = {distances[i]:.2f})")
            continue
        filtered_matches.append(m)

    return filtered_matches



def estimate_pose(real_img_path, reference_img_path, json_path, K, dist_coeffs=None, max_keypoints=50, compute_ref_pose=False, visualize=False):
    
        
    real_img = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)
    reference_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
    reference_filename = reference_img_path.split("\\")[-1]

    reference_img_coords, reference_obj_coords = load_reference_keypoints(json_path, reference_filename)

    real_coords = run_detection_pipeline(real_img_path, max_corners=20, visualize=False)["corner_coords"]

    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    kp_real = [cv2.KeyPoint(float(x), float(y), 15) for (x, y) in real_coords]
    kp_reference = [cv2.KeyPoint(float(x), float(y), 15) for (x, y) in reference_img_coords]

    _, des_real = sift.compute(real_img, kp_real)
    _, des_reference = sift.compute(reference_img, kp_reference)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_reference, des_real, k=2)

    ratio = 0.95
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    good_matches = filter_matches_by_best_proximity(good_matches, kp_reference, kp_real, max_pixel_dist=1000)
    # Additional culling: Remove matches with high pixel discrepancy (z-score outliers)
    good_matches = remove_pixel_outliers_by_zscore(good_matches, kp_reference, kp_real, z_thresh=3)
    

    matched_3D = [reference_obj_coords[m.queryIdx] for m in good_matches]
    matched_real_2D = [kp_real[m.trainIdx].pt for m in good_matches]
    matched_ref_2D = [kp_reference[m.queryIdx].pt for m in good_matches]

    rvec_real, tvec_real = None, None
    if len(matched_3D) >= 4:
        success, rvec_real, tvec_real, inliers = cv2.solvePnPRansac(np.array(matched_3D, dtype=np.float64),
                                                     np.array(matched_real_2D, dtype=np.float64),
                                                     K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP, reprojectionError=10, confidence=0.99, iterationsCount=100)
        if success:
            print("Pose estimation with solvePnPRansac successful for new image.")
        else:
            print("Pose estimation failed for new image.")
    else:
        print("Not enough matches for solvePnP on new image.")

    

    if not success:
        print(f"Reattempting pose estimation by culling down from {len(matched_3D)} 2D–3D matches:")
    
        rvec_real, tvec_real, inliers = None, None, None
        max_attempts = len(good_matches) - 5  # Stop at 5 matches
    
        for attempt in range(max_attempts):
            # Remove worst match based on error between feature point locations  in image
            good_matches = remove_worst_pixel_match(good_matches, kp_reference, kp_real)
            matched_3D = [reference_obj_coords[m.queryIdx] for m in good_matches]
            matched_real_2D = [kp_real[m.trainIdx].pt for m in good_matches]
    
            if len(matched_3D) < 4:
                print("Not enough points remain after culling for solvePnP.")
                break
    
            # Reattempt pose estimation
            success_retry, rvec_attempt, tvec_attempt, inliers_attempt = cv2.solvePnPRansac(
                np.array(matched_3D, dtype=np.float64),
                np.array(matched_real_2D, dtype=np.float64),
                K, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=10, confidence=0.99, iterationsCount=100
            )
    
            print(f"[Retry {attempt+1}] Matches remaining: {len(matched_3D)}")
    
            if success_retry:
                print("Pose estimation succeeded after culling.")
                rvec_real, tvec_real, inliers = rvec_attempt, tvec_attempt, inliers_attempt
                success = True
                break
            else:
                print("Still failed.")
    
        if not success:
            print("Final pose estimation failed after all culling attempts.")
    

    # Refinement using inliers
    if success and inliers is not None and len(inliers) >= 4:
        inlier_obj_pts = np.array(matched_3D, dtype=np.float64)[inliers.ravel()]
        inlier_img_pts = np.array(matched_real_2D, dtype=np.float64)[inliers.ravel()]
    
        success_refine, rvec_refined, tvec_refined = cv2.solvePnP(
            inlier_obj_pts,
            inlier_img_pts,
            K.astype(np.float64),
            dist_coeffs.astype(np.float64),
            rvec_real,
            tvec_real,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    
        if success_refine:
            rvec_real, tvec_real = rvec_refined, tvec_refined
            print("Pose refined with solvePnP (ITERATIVE).")
        else:
            print("Pose refinement failed.")


    
    rvec_ref, tvec_ref = None, None

    if compute_ref_pose:
        # Solve for reference pose using 3D CAD points and their 2D reference image points
        if len(reference_obj_coords) >= 4 and len(reference_img_coords) >= 4:
            success_ref, rvec_ref, tvec_ref, inliers = cv2.solvePnPRansac(np.array(reference_obj_coords, dtype=np.float64),
                                                          np.array(reference_img_coords, dtype=np.float64),
                                                          K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP, reprojectionError=10, confidence=0.99, iterationsCount=100)
            if success_ref:
                print("Pose estimation with solvePnPRansac successful for reference image.")
            else:
                print("Pose estimation failed for reference image.")
        else:
            print("Not enough points for solvePnP on reference image.")
            
            
        # Optional refinement using inliers
        if success and inliers is not None and len(inliers) >= 4:
            inlier_obj_pts = np.array(reference_obj_coords, dtype=np.float64)[inliers.ravel()]
            inlier_img_pts = np.array(reference_img_coords, dtype=np.float64)[inliers.ravel()]
        
            success_refine, rvec_refined, tvec_refined = cv2.solvePnP(
                inlier_obj_pts,
                inlier_img_pts,
                K.astype(np.float64),
                dist_coeffs.astype(np.float64),
                rvec_ref,
                tvec_ref,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        
            if success_refine:
                rvec_ref, tvec_ref = rvec_refined, tvec_refined
                print("Pose refined with solvePnP (ITERATIVE).")
            else:
                print("Pose refinement failed.")
            
            

    if visualize:
        plot_keypoints_and_matches(real_img, reference_img, kp_real, kp_reference, good_matches)

            

    return rvec_real, tvec_real, matched_real_2D, matched_ref_2D, rvec_ref, tvec_ref






"""
Main Function Summary
---------------------

This script performs end-to-end object pose estimation and robot pose correction using a monocular camera setup.
It executes the following steps:
1. Loads reference 2D–3D keypoint correspondences from a JSON file.
2. Detects corners in a real image using a machine learning model, and extracts SIFT features from both real and reference images.
3. Matches keypoints and estimates object pose in the real image using solvePnPRansac, followed by refinement with solvePnP (ITERATIVE).
4. Optionally computes and refines the pose of the object in the reference image using its annotated 2D–3D pairs.
5. Converts the rvec and tvec of both images into human-readable 6DOF poses.
6. Uses a known hand–eye calibration matrix (T_cam_tool) to update and correct the test robot tool pose in base frame.
7. Visualizes and compares the original, reference, and corrected tool poses.
"""

if __name__ == "__main__":
    
    #===============USER CONFIG====================
    # Define filepaths
    real_img_path = r"Path to new image"
    reference_img_path = r"Path to reference image"
    json_path = r"Path to feature point 2D-3D mapping .json file"


    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)


    rvec_real, tvec_real, matched_real_2D, matched_ref_2D, rvec_ref, tvec_ref = estimate_pose(
        real_img_path, reference_img_path, json_path, K, dist_coeffs,
        compute_ref_pose=True, visualize=True
    )
    

    print("Matched Real Image Points:", matched_real_2D)
    print("Matched Reference Image Points:", matched_ref_2D)
    


    
    if all(x is not None for x in [rvec_real, tvec_real, rvec_ref, tvec_ref]):
        pose_real = rvec_tvec_to_pose6d(rvec_real, tvec_real)
        pose_ref  = rvec_tvec_to_pose6d(rvec_ref,  tvec_ref)
        # delta     = [ref - real for ref, real in zip(pose_ref, pose_real)]
    
        labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    
        print("\n=== Object Pose relative to Camera (Real) ===")
        for lbl, val in zip(labels, pose_real):
            print(f"{lbl} = {val:.2f}")
    
        print("\n=== Object Pose relative to Camera (Reference) ===")
        for lbl, val in zip(labels, pose_ref):
            print(f"{lbl} = {val:.2f}")
    
    else:
        print("Pose estimation failed for at least one image.")

        


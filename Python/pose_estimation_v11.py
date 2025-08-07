import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from ML_Segmentation_v25 import run_detection_pipeline  



"""
Function Summaries
------------------

load_reference_keypoints(json_path, image_name)
    Loads 2D keypoints from a JSON file for a given reference image and matches them to their corresponding 3D CAD points using an ID lookup.

filter_matches_by_best_proximity(good_matches, kp1, kp2, max_pixel_dist=None)
    Filters SIFT feature matches to retain only mutual best matches within an optional pixel distance threshold. Helps reject ambiguous or one-way matches.

plot_keypoints_and_matches(real_img, reference_img, kp_real, kp_reference, good_matches)
    Draws annotated keypoints on both the real and reference images, coloring matched keypoints green and unmatched ones blue, then displays them side by side.

rvec_tvec_to_pose6d(rvec, tvec)
    Converts OpenCV rotation (rvec) and translation (tvec) vectors into a 6-DOF pose represented as [x, y, z, roll, pitch, yaw] with rotation in degrees.

estimate_pose(real_img_path, reference_img_path, json_path, K, dist_coeffs=None, max_keypoints=50, compute_ref_pose=False, visualize=False)
    Performs keypoint detection, SIFT descriptor matching, and 6DOF pose estimation using solvePnP on the real image.
    Optionally also estimates the reference image pose for comparison.
    Returns rvec and tvec for both images (if computed), along with the matched 2D coordinates.
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
    plt.show()

    

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

    matched_3D = [reference_obj_coords[m.queryIdx] for m in good_matches]
    matched_real_2D = [kp_real[m.trainIdx].pt for m in good_matches]
    matched_ref_2D = [kp_reference[m.queryIdx].pt for m in good_matches]

    rvec_real, tvec_real = None, None
    if len(matched_3D) >= 4:
        success, rvec_real, tvec_real, inliers = cv2.solvePnPRansac(np.array(matched_3D, dtype=np.float32),
                                                     np.array(matched_real_2D, dtype=np.float32),
                                                     K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP, reprojectionError=10, confidence=0.99, iterationsCount=100)
        if success:
            print("Pose estimation successful.")
        else:
            print("Pose estimation failed for real image.")
    else:
        print("Not enough matches for solvePnP on real image.")

    rvec_ref, tvec_ref = None, None
    if compute_ref_pose:
        # Solve for reference pose using 3D CAD points and their 2D reference image points
        if len(reference_obj_coords) >= 4 and len(reference_img_coords) >= 4:
            success_ref, rvec_ref, tvec_ref = cv2.solvePnP(np.array(reference_obj_coords, dtype=np.float32),
                                                          np.array(reference_img_coords, dtype=np.float32),
                                                          K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            if success_ref:
                print("Pose estimation successful.")
            else:
                print("Pose estimation failed for reference image.")
        else:
            print("Not enough points for solvePnP on reference image.")
            
                   
    if visualize:
        plot_keypoints_and_matches(real_img, reference_img, kp_real, kp_reference, good_matches)


    return rvec_real, tvec_real, matched_real_2D, matched_ref_2D, rvec_ref, tvec_ref






"""
---------------------
Main Function Summary
---------------------

This script estimates the 6DOF pose of a known object in a real image using manually annotated keypoints from a reference image.
It performs the following steps:
1. Loads and matches 2D–3D keypoints for the reference image using a JSON file.
2. Detects keypoints in a real image using a machine learning-based corner detection pipeline.
3. Computes feature descriptors and performs SIFT matching between real and reference keypoints.
4. Filters matches and estimates object pose in the real image using solvePnPRansac.
5. Optionally computes object pose in the reference image.
6. Converts both poses into 6DOF format (x, y, z, roll, pitch, yaw) and displays them.
7. Uses a hand–eye calibration matrix (T_cam_tool) to correct the test robot tool pose and visualize differences between reference, test, and corrected poses.
"""


if __name__ == "__main__":
    
    #==============USER CONFIG==================
    # Define filepaths
    real_img_path = r"Path to new image"
    reference_img_path = r"Path to reference image"
    json_path = r"Path to .json file holding 2D-3D keypoint mapping"

    # Define camera intrinsics
    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    # Define 6D tool pose for each image as [x, y, z, roll, pitch, yaw]
    test_tool_pose_base = [0, 0, 0, 0, 0, 0]  # mm and degrees. 
    ref_tool_pose_base = [0, 0, 0, 0, 0, 0]
    #=============================================



    rvec_real, tvec_real, matched_real_2D, matched_ref_2D, rvec_ref, tvec_ref = estimate_pose(
        real_img_path, reference_img_path, json_path, K, dist_coeffs,
        compute_ref_pose=True, visualize=True
    )
    
    
    print("Matched Real Image Points:", matched_real_2D)
    print("Matched Reference Image Points:", matched_ref_2D)
        
    # from Pose_Interpretation_v13 import explain_pose
    # explain_pose(rvec_real, tvec_real, Verbose=False)


    if all(x is not None for x in [rvec_real, tvec_real, rvec_ref, tvec_ref]):
        pose_real = rvec_tvec_to_pose6d(rvec_real, tvec_real)
        pose_ref  = rvec_tvec_to_pose6d(rvec_ref,  tvec_ref)
    
        labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    
        print("\n=== Object Pose relative to Camera (Real) ===")
        for lbl, val in zip(labels, pose_real):
            print(f"{lbl} = {val:.2f}")
    
        print("\n=== Object Pose relative to Camera (Reference) ===")
        for lbl, val in zip(labels, pose_ref):
            print(f"{lbl} = {val:.2f}")
    
    else:
        print("Pose estimation failed for at least one image.")

        




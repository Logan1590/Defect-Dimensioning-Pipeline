import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import time

from Eye_in_Hand_Calibration_v4 import retrieve_eyehand_calibration
from Image_Comparison_v2 import process_image_differences
from pose_estimation_v12 import estimate_pose, rvec_tvec_to_pose6d


"""
Function Summaries
------------------

rpy_to_rotation_matrix(roll, pitch, yaw):
    - Converts roll, pitch, and yaw angles (in degrees) into a 3×3 rotation matrix using OpenCV Rodrigues.

pose_to_matrix(xyzrpy):
    - Converts a 6DOF pose [x, y, z, roll, pitch, yaw] to a 4×4 homogeneous transformation matrix.

matrix_to_pose(T):
    - Converts a 4×4 homogeneous transformation matrix into a 6DOF pose vector [x, y, z, roll, pitch, yaw].

compute_and_plot_diff_heatmap(ref_image_path, test_image_path, corrected_image_path):
    - Computes and displays grayscale image differences and heatmaps between:
        (a) reference and test image,
        (b) reference and corrected image.

plot_three_poses(ref_tool_pose_base, test_tool_pose_base, corrected_tool_pose_base):
    - 3D visualization of the reference, test, and corrected tool poses in space.
    - Displays orientation using XYZ axes and labels.

rotation_translation_error(pose_est, pose_gt):
    - Calculates rotation (degrees) and translation (mm) error between two 6DOF poses.

compare_pose_differences(ref_tool_pose_base, corrected_tool_pose_base, test_tool_pose_base, printout=False):
    - Compares the corrected pose and test pose to the reference pose.
    - Computes absolute and percent improvement in translation and rotation errors.
    - Returns a dictionary of all errors and improvements.

compute_reference_tool_pose(test_tool_pose_base, object_pose_test_cam, object_pose_ref_cam, T_cam_tool):
    - Uses pose chaining and eye-in-hand calibration to estimate the tool pose in base frame
      that would generate the reference camera view.
    - Returns corrected tool pose in base frame.
"""


    
def rpy_to_rotation_matrix(roll, pitch, yaw):
    """Convert roll, pitch, yaw (in degrees) to a rotation matrix."""
    rvec = np.deg2rad([roll, pitch, yaw])
    rvec = rvec.reshape((3, 1))
    R, _ = cv2.Rodrigues(rvec)
    return R



def pose_to_matrix(xyzrpy):
    """Convert 6DOF pose to matrix"""
    t = np.array(xyzrpy[:3])
    r = R.from_euler('xyz', xyzrpy[3:], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T



def matrix_to_pose(T):
    """Convert matrix to 6DOF pose"""
    t = T[:3, 3]
    r = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return np.concatenate([t, r])



def compute_and_plot_diff_heatmap(ref_image_path, test_image_path, corrected_image_path):
    # Load images as grayscale for pixel difference
    img1 = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(corrected_image_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None or img3 is None:
        raise FileNotFoundError("One or more image paths are invalid or files are not readable.")

    # Ensure all images are same shape
    if img1.shape != img2.shape or img1.shape != img3.shape:
        raise ValueError("All images must be the same dimensions.")
        
    
    # === Plot 1x3: Original images ===
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    axs1[0].imshow(img1, cmap='gray')
    axs1[0].set_title("Reference Perspective")
    axs1[1].imshow(img2, cmap='gray')
    axs1[1].set_title("Test Perspective")
    axs1[2].imshow(img3, cmap='gray')
    axs1[2].set_title("Corrected Perspective")

    for ax in axs1:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    

    # Compute absolute difference heatmaps
    diff1 = cv2.absdiff(img1, img2)
    diff2 = cv2.absdiff(img1, img3)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(diff1, cmap='hot')
    axs[0].set_title('|Reference - Test|')
    axs[0].axis('off')

    axs[1].imshow(diff2, cmap='hot')
    axs[1].set_title('|Reference - Corrected|')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()



def plot_three_poses(ref_tool_pose_base, test_tool_pose_base, corrected_tool_pose_base, axis_length=50):
    
    def plot_pose(ax, pose, axis_length=50, label_prefix="Pose", show_legend=False):
        x, y, z, roll, pitch, yaw = pose
        origin = np.array([x, y, z])
        R = rpy_to_rotation_matrix(roll, pitch, yaw)

        # Endpoints of each axis vector
        x_axis = origin + R[:, 0] * axis_length
        y_axis = origin + R[:, 1] * axis_length
        z_axis = origin + R[:, 2] * axis_length

        # Plot arrows (Z-axis gets label once for legend)
        ax.quiver(*origin, *(x_axis - origin), color='r')
        ax.quiver(*origin, *(y_axis - origin), color='g')
        ax.quiver(*origin, *(z_axis - origin), color='b', label="Camera View" if show_legend else None)

        # Add label near origin
        ax.text(origin[0], origin[1], origin[2] + axis_length * 0.2, label_prefix,
                color='black', fontsize=16, horizontalalignment='center')

    named_poses = [
        ("Reference", ref_tool_pose_base),
        ("Test", test_tool_pose_base),
        ("Corrected", corrected_tool_pose_base)
    ]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, (name, pose) in enumerate(named_poses):
        plot_pose(ax, pose, axis_length=axis_length, label_prefix=name, show_legend=(i == 0))

    # Set plot limits
    all_origins = np.array([pose[:3] for _, pose in named_poses])
    center = np.mean(all_origins, axis=0)
    lim = axis_length * 3
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_zlim(center[2] - lim, center[2] + lim)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Comparison of 3 Tool Poses")
    ax.view_init(elev=30, azim=120)

    # Add legend
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    
    
    
def rotation_translation_error(pose_est, pose_gt):
    """Compute rotation (°) and translation (mm) error between two 6DOF poses."""
    T_est = pose_to_matrix(pose_est)
    T_gt = pose_to_matrix(pose_gt)

    delta_T = np.linalg.inv(T_gt) @ T_est
    delta_t = delta_T[:3, 3]
    delta_R = delta_T[:3, :3]
    rot_error = np.rad2deg(np.arccos(np.clip((np.trace(delta_R) - 1) / 2, -1, 1)))

    trans_error = np.linalg.norm(delta_t)
    return trans_error, rot_error
    


def compare_pose_differences(ref_tool_pose_base, corrected_tool_pose_base, test_tool_pose_base, printout=False):
    """
    Compares 6DOF poses and reports:
    - Rotation and translation errors between all pairs
    - Improvement from test to corrected
    - Percent improvement relative to reference pose
    """

    # Compute errors
    trans_test, rot_test = rotation_translation_error(test_tool_pose_base, ref_tool_pose_base)
    trans_corr, rot_corr = rotation_translation_error(corrected_tool_pose_base, ref_tool_pose_base)

    # Compute improvements
    trans_improv = trans_test - trans_corr
    rot_improv = rot_test - rot_corr

    # Compute percent improvement (guarding against division by zero)
    trans_pct = (trans_improv / trans_test * 100) if trans_test != 0 else 0
    rot_pct = (rot_improv / rot_test * 100) if rot_test != 0 else 0

    if printout:
        print('=== Tool Poses ===')
        print('Reference Image', [round(num,2) for num in ref_tool_pose_base])
        print('Test Image     ', [round(num,2) for num in test_tool_pose_base])
        print('Corrected Image', [round(num,2) for num in corrected_tool_pose_base])
        
        print("\n=== Tool Pose Differences ===")
        print(f"Test     → Reference:  ΔT = {trans_test:.2f} mm, ΔR = {rot_test:.2f}°")
        print(f"Corrected→ Reference:  ΔT = {trans_corr:.2f} mm, ΔR = {rot_corr:.2f}°")
        print("\n=== Improvement (Corrected vs Test) ===")
        print(f"Translation Improvement: {trans_improv:.2f} mm ({trans_pct:.1f}%)")
        print(f"Rotation Improvement:    {rot_improv:.2f}°  ({rot_pct:.1f}%)")

    return {
        "translation_test": trans_test,
        "rotation_test": rot_test,
        "translation_corrected": trans_corr,
        "rotation_corrected": rot_corr,
        "translation_improvement": trans_improv,
        "rotation_improvement": rot_improv,
        "translation_percent": trans_pct,
        "rotation_percent": rot_pct
    }


            
def compute_reference_tool_pose(test_tool_pose_base, object_pose_test_cam, object_pose_ref_cam, T_cam_tool=np.eye(4)):
    """
    Given:
    - test_tool_pose_base: known pose of tool in base frame for the test image [x,y,z,roll,pitch,yaw]
    - object_pose_test_cam: pose of object as seen from test camera view [x,y,z,roll,pitch,yaw]
    - object_pose_ref_cam: pose of object as seen from reference camera view [x,y,z,roll,pitch,yaw]
    - T_cam_tool: eye-hand calibration transform from camera to tool (identity if camera = tool)

    Returns:
    - corrected_tool_pose_base: estimated pose of tool in base frame for the reference image
    """
    T_tool_base_test = pose_to_matrix(test_tool_pose_base)
    T_cam_base_test = T_tool_base_test @ T_cam_tool

    T_obj_cam_test = pose_to_matrix(object_pose_test_cam)
    T_obj_base = T_cam_base_test @ T_obj_cam_test
    

    T_obj_cam_ref = pose_to_matrix(object_pose_ref_cam)
    T_cam_base_ref = T_obj_base @ np.linalg.inv(T_obj_cam_ref)
    

    T_tool_base_ref = T_cam_base_ref @ np.linalg.inv(T_cam_tool)
    
    corrected_tool_pose_base = matrix_to_pose(T_tool_base_ref)
    corrected_tool_pose_base = [float(round(val, 2)) for val in corrected_tool_pose_base]
    
    
    return corrected_tool_pose_base





"""

Main Function Summary
---------------------
- Loads test and reference image paths and estimates the 6DOF object pose in each camera view using 2D–3D keypoints.
- Computes the corrected tool pose that would bring the camera into the reference view using eye-in-hand calibration.
- Compares the reference, test, and corrected tool poses:
    - Visualizes them in 3D.
    - Quantifies improvement in translation and rotation.
- Optionally, loads the image taken from the corrected pose and computes visual difference heatmaps between views.
"""

if __name__ == "__main__":
    
    start_time = time.perf_counter()

    
    #======================USER CONFIG========================
    # Define filepaths
    test_image = r"Path to new image"    
    ref_image = r"Path to reference image"
    json_keypoints_path = r"Path to 2D-3D keypoint mapping .json file"

    # Camera intrinsics (example values)
    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    # Define 6D tool pose for test and reference images as [X, Y, Z, roll, pitch, yaw] 
    ref_tool_pose_base = [0, 0, 0, 0, 0, 0]
    test_tool_pose_base = [0, 0, 0, 0, 0, 0]  # mm and degrees. 
    #===========================================================
    
    
    
    rvec, tvec, _, _, rvec_ref, tvec_ref = estimate_pose(
            test_image,
            ref_image,
            json_keypoints_path,
            K,
            dist_coeffs,
            compute_ref_pose=True,
            visualize=False
        )
    
    object_pose_test_cam = rvec_tvec_to_pose6d(rvec, tvec)
    object_pose_ref_cam = rvec_tvec_to_pose6d(rvec_ref, tvec_ref)
        

    
    # Retrieve the eye_hand calibration matrix calculated separately
    T_cam_tool = retrieve_eyehand_calibration()
        
    
    # Find the tool position that returns the camera to the reference view
    corrected_tool_pose_base = compute_reference_tool_pose(
        test_tool_pose_base,
        object_pose_test_cam,
        object_pose_ref_cam,
        T_cam_tool
    )
    
    
    end_time = time.perf_counter()
    print(f"Time to estimate perspective correction: {end_time - start_time:.4f} seconds")
    
    
     
    # Plot the orientation of the tool in each pose (inital reference to match, test image after rough alignment, corrected position), 
    plot_three_poses(ref_tool_pose_base, test_tool_pose_base, corrected_tool_pose_base)
    results = compare_pose_differences(ref_tool_pose_base, corrected_tool_pose_base, test_tool_pose_base, printout=True)
        

    
    # Visualization: If you have the image from the corrected tool position, display heat maps of the differences:
    corrected_image = r"Path to image taken from corrected position"
    compute_and_plot_diff_heatmap(ref_image, test_image, corrected_image)
    diff, binary, mask = process_image_differences(test_image, corrected_image)

    
    

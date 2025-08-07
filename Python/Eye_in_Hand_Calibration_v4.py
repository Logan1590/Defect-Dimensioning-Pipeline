import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import re
from scipy.spatial.transform import Rotation as R




"""
Function Summaries
------------------

natural_key(p):
    - Returns a natural sort key from a filename, splitting digits from text to allow human-style ordering (e.g., image2 < image10).

rvec_to_z_axis(rvec):
    - Converts a rotation vector to a rotation matrix and returns the Z-axis direction (third column) in camera coordinates.

draw_pose_axes_matplotlib(image, corners, K, dist_coeffs, rvec, tvec, axis_length):
    - Projects and draws 3D coordinate axes (X: red, Y: green, Z: blue) onto an image at the checkerboard origin.
    - Uses matplotlib to overlay pose and detected corners with labels.

rpy_to_matrix(roll, pitch, yaw):
    - Converts roll, pitch, and yaw (in degrees) to a 3x3 rotation matrix using scipy’s `Rotation.from_euler`.

retrieve_eyehand_calibration():
    - Returns a hardcoded 4x4 identity matrix as a placeholder for camera-to-tool transformation (T_cam_tool).
"""



def natural_key(p: Path):
    # Split the filename into text and number parts
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', p.stem)]



def rvec_to_z_axis(rvec):
    """Returns the Z-axis direction of the object (checkerboard) in the camera frame."""
    R, _ = cv2.Rodrigues(rvec)
    return R[:, 2]  # Z-axis of the object in camera coordinates



def draw_pose_axes_matplotlib(image, corners, K, dist_coeffs, rvec, tvec, axis_length=50):
    """Draw coordinate axes of checkerboard on checkerboard images."""

    
    if image is None:
        print("[ERROR] Image is None — skipping plot.")
        return
    if corners is None or len(corners) == 0:
        print("[WARNING] Corners are empty — skipping plot.")
        return

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()

        # Project 3D axes
        axes_3d = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ])
        imgpts, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist_coeffs)

        if imgpts is None or np.isnan(imgpts).any():
            print("[WARNING] Invalid projection points — skipping plot.")
            return
        imgpts = imgpts.reshape(-1, 2)
        origin, x_axis, y_axis, z_axis = imgpts

        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)

        try:
            if corners.ndim == 3 and corners.shape[2] == 2:
                x = corners[:, 0, 0]
                y = corners[:, 0, 1]
            else:
                x = corners[:, 0]
                y = corners[:, 1]
            plt.plot(x, y, 'o', color='cyan', label='Corners')
        except Exception as e:
            print(f"[WARNING] Corner plotting failed: {e}")

        plt.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], color='red', label='X axis')
        plt.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], color='green', label='Y axis')
        plt.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], color='blue', label='Z axis')

        plt.title("Checkerboard Pose Visualization")
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[ERROR] draw_pose_axes_matplotlib failed: {e}")



def rpy_to_matrix(roll, pitch, yaw):
    """Convert roll, pitch, yaw in degrees to a rotation matrix."""
    
    return R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()



def retrieve_eyehand_calibration():
    """
    Return a hardcoded 4x4 camera-to-end-effector transformation matrix
    obtained from a prior hand-eye calibration, for use in other scripts.
    """

    T_cam_tool = np.array( 
    [[1,  0, 0,  0],
    [0, 1,  0, 0],
    [0, 0, 1, 0],
    [ 0, 0, 0, 1]]
     )

    return T_cam_tool





"""
Main Function Summary
---------------------
- Loads a series of checkerboard images and corresponding 6D tool poses in base frame.
- Detects checkerboard corners in each image and estimates the camera-to-checkerboard pose using `solvePnP`.
- Projects and visualizes the pose axes on each image to verify pose accuracy.
- Constructs base-to-tool and camera-to-target transformations.
- Uses OpenCV’s `calibrateHandEye` to compute the 4x4 homogeneous camera-to-tool transformation matrix.
- Outputs the resulting hand-eye calibration result (rotation, translation, and full matrix).
"""

if __name__ == "__main__":
    
    #==============USER CONFIG==================
    
    # === Checkerboard parameters ===
    CHECKERBOARD = (10, 7)   # of checker corners (row, column)
    SQUARE_SIZE = 15  # mm

    # === Camera intrinsics and distortion ===
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    
    
    # Define filepaths
    image_dir = Path(r"Path to checkerboard images")

    
    # Provide 6D pose (X, Y, Z, roll, pitch, yaw) of tool that camera is mounted on for each image in base world frame
    tool_poses_rpy = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    #=============================================
    


    R_tool2base = []
    t_tool2base = []
    for pose in tool_poses_rpy:
        x, y, z, roll, pitch, yaw = pose
        R = rpy_to_matrix(roll, pitch, yaw)
        t = np.array([[x], [y], [z]])
        R_tool2base.append(R)
        t_tool2base.append(t)



    # === Prepare 3D object points ===
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        

    objp *= SQUARE_SIZE



    # === Load images ===

    images = sorted(image_dir.glob("*.jpg"), key=natural_key)

    R_target2cam = []
    t_target2cam = []

    for img_path in images:
        print('Image path:', img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] Failed to load image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
           
            
            # Get full pose from solvePnP
            success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist_coeffs)
            if success:
                R_cam, _ = cv2.Rodrigues(rvec)
                z_axis = R_cam[:, 2]  # Z-axis of checkerboard in camera frame
            
                R_target2cam.append(R_cam)
                t_target2cam.append(tvec)
                print(f"[INFO] Pose found for {img_path.name}.")
                draw_pose_axes_matplotlib(img, corners, K, dist_coeffs, rvec, tvec)

                z_axis = rvec_to_z_axis(rvec)
                print(f"       rvec = {rvec.ravel()}, tvec = {tvec.ravel()}")
                print(f"       Z-axis of checkerboard in camera frame: {z_axis} (should be ~[0, 0, 1])")
            else:
                print(f"[WARN] solvePnP failed on {img_path.name}")
        else:
            print(f"[WARN] Checkerboard not found in {img_path.name}")

    if len(R_target2cam) != len(R_tool2base):
        raise ValueError("Mismatch in number of camera poses and tool poses!")



    # === Perform hand-eye calibration ===
    R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
        R_tool2base, t_tool2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2tool = np.eye(4)
    T_cam2tool[:3, :3] = R_cam2tool
    T_cam2tool[:3, 3] = t_cam2tool.ravel()

    print("\n=== Hand-Eye Calibration Result ===")
    print("Rotation Matrix:\n", R_cam2tool)
    print("Translation Vector:\n", t_cam2tool.ravel())
    print("Homogeneous Matrix:\n", T_cam2tool)




import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os




"""
Function Summaries
------------------

calibrate_camera_from_checkerboard_folder(folder_path, checkerboard_dims, square_size):
    - Detects checkerboard corners in a folder of images and calibrates the camera.
    - Returns calibration success flag, intrinsic camera matrix, distortion coefficients, and image size.

undistort_image(image_path, camera_matrix, dist_coeffs):
    - Loads an image and applies undistortion using the given camera intrinsics.
    - Returns both the original and undistorted images (same dimensions, no cropping).

show_images(original, corrected, titles):
    - Displays the original and undistorted images side by side using matplotlib.
    - Converts BGR to RGB and applies consistent axis formatting and aspect ratio.
"""



def calibrate_camera_from_checkerboard_folder(folder_path, checkerboard_dims=(9, 6), square_size=1.0):
    """
    Calibrate camera using multiple checkerboard images in a folder.

    Parameters:
        folder_path (str): Path to folder with checkerboard images
        checkerboard_dims (tuple): Number of inner corners per row and column (width, height)
        square_size (float): Size of one square in your desired unit (e.g. mm or cm)

    Returns:
        ret: Calibration success flag
        camera_matrix: Intrinsic matrix
        dist_coeffs: Distortion coefficients
        img_shape: Shape of the calibration images
    """
    objp = np.zeros((checkerboard_dims[1]*checkerboard_dims[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))

    if not images:
        raise RuntimeError(f"No images found in {folder_path}")

    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            print(f"Checkerboard not detected in {fname}")

    if len(objpoints) == 0:
        raise RuntimeError("No checkerboard corners detected in any image.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None)

    return ret, camera_matrix, dist_coeffs, img_shape



def undistort_image(image_path, camera_matrix, dist_coeffs):
    img = cv2.imread(image_path)
    # Undistort using original camera matrix directly (no scaling or cropping)
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)

    return img, undistorted_img  # no cropping, original size



def show_images(original, corrected, titles=("Original", "Undistorted")):
    plt.figure(figsize=(12, 5))
    for i, (img, title) in enumerate(zip([original, corrected], titles)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, i + 1)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
        plt.gca().set_aspect('equal')  # ← important!
    plt.tight_layout()
    plt.show()



"""
Main Function Summary
---------------------
- Loads a folder of checkerboard images and calibrates the camera based on corner detection.
- Computes focal lengths in millimeters and image center shifts using the calibrated intrinsics.
- Undistorts a target image using the derived camera matrix and distortion coefficients.
- Displays the original and corrected image side by side for visual comparison.
"""



if __name__ == "__main__":
    
    # ============= USER CONFIG ==============
    checkerboard_folder = r'Path to checkerboard images folder'
    image_to_undistort = r'Path to image to undistort'
    #=========================================


    # === Step 1: Calibrate ===
    # Make sure to update checkernoard_dims to hold the # of checker corners in each row and column, and the size of each checker square
    ret, cam_matrix, dist_coeffs, img_shape = calibrate_camera_from_checkerboard_folder(
        checkerboard_folder, checkerboard_dims=(10, 7), square_size=10)

    print("Camera matrix:\n", cam_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())

    # === Step 2: Extract Intrinsics ===
    fx = cam_matrix[0, 0]
    fy = cam_matrix[1, 1]
    cx = cam_matrix[0, 2]
    cy = cam_matrix[1, 2]

    image_width, image_height = img_shape  # (width, height)
    sensor_width_mm = 'Insert width in mm'
    sensor_height_mm = 'Insert width in mm'

    # Compute focal length in mm
    focal_length_mm_x = fx * sensor_width_mm / image_width
    focal_length_mm_y = fy * sensor_height_mm / image_height

    # Compute Blender-style shift values
    shift_x = (cx - image_width / 2) / image_width
    shift_y = (cy - image_height / 2) / image_height

    print("\n--- Camera Parameters ---")
    print(f"Image size: {image_width} x {image_height} pixels")
    print(f"Sensor size: {sensor_width_mm:.2f} x {sensor_height_mm:.2f} mm")
    print(f"Focal length (X): {focal_length_mm_x:.4f} mm")
    print(f"Focal length (Y): {focal_length_mm_y:.4f} mm")
    print(f"Shift X: {shift_x:.6f}")
    print(f"Shift Y: {shift_y:.6f}")

    # === Step 3: Undistort and Show ===
    original, corrected = undistort_image(image_to_undistort, cam_matrix, dist_coeffs)
    show_images(original, corrected)



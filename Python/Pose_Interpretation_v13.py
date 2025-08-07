import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pose_estimation_v12 import estimate_pose


"""
Function Summaries
------------------

explain_pose(rvec, tvec, rvec_ref=None, tvec_ref=None, Verbose=True):
    - Prints and visualizes the 6DOF pose of an object (translation + rotation).
    - If a reference pose is provided, it also prints the delta between estimated and reference.
    - Visualizes a top-down view of the scene with object footprint and axes.

draw_translation_annotations_with_scale(img, rvec, tvec, K, dist_coeffs):
    - Annotates the input image with CAD origin and object translation vectors (+X and +Y).
    - Adds scale axes to the plot, where the origin is the image center in camera coordinates.
    - Displays roll, pitch, yaw along with physical position.

draw_translation_annotations(img, rvec, tvec, K, dist_coeffs):
    - Projects the CAD origin and object axes onto the image.
    - Draws thick arrows for X and Y directions from the image center to the object origin.
    - Labels the 3D object coordinate axes with X, Y, Z colors and subscripts.
"""


def explain_pose(rvec, tvec, rvec_ref=None, tvec_ref=None, Verbose=True):
    def get_pose_data(rvec, tvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        tvec = tvec.reshape(3, 1)
        return {
            "rvec": rvec,
            "tvec": tvec,
            "R": rotation_matrix,
            "tx": tvec[0, 0],
            "ty": tvec[1, 0],
            "tz": tvec[2, 0],
            "roll": euler_angles[0],
            "pitch": euler_angles[1],
            "yaw": euler_angles[2],
        }

    pose = get_pose_data(rvec, tvec)
    
    if Verbose:
        print("\n=== Interpreting Estimated Pose ===\n")
        print("Rotation Matrix (R):", pose["R"])
        print("\nTranslation Vector (tvec):", pose["tvec"].ravel())
        print(f"""
        Interpretation in the Image:
        - X (right/left): {pose['tx']:.2f} mm
        - Y (down/up): {pose['ty']:.2f} mm
        - Z (forward): {pose['tz']:.2f} mm
        - Roll (X): {pose['roll']:.2f} deg
        - Pitch (Y): {pose['pitch']:.2f} deg
        - Yaw (Z): {pose['yaw']:.2f} deg
        """)

    if rvec_ref is not None and tvec_ref is not None:
        ref_pose = get_pose_data(rvec_ref, tvec_ref)
        
        if Verbose:
            print("\n=== Reference Pose ===\n")
            print("Rotation Matrix (R):", ref_pose["R"])
            print("\nTranslation Vector (tvec):", ref_pose["tvec"].ravel())
            print(f"""
            Reference Pose:
            - X: {ref_pose['tx']:.2f} mm
            - Y: {ref_pose['ty']:.2f} mm
            - Z: {ref_pose['tz']:.2f} mm
            - Roll: {ref_pose['roll']:.2f} deg
            - Pitch: {ref_pose['pitch']:.2f} deg
            - Yaw: {ref_pose['yaw']:.2f} deg
            """)

        dx = pose["tx"] - ref_pose["tx"]
        dy = pose["ty"] - ref_pose["ty"]
        dz = pose["tz"] - ref_pose["tz"]
        droll = pose["roll"] - ref_pose["roll"]
        dpitch = pose["pitch"] - ref_pose["pitch"]
        dyaw = pose["yaw"] - ref_pose["yaw"]
        
        if Verbose:
            print("=== Pose Differences in Image Axes (Estimated - Reference) ===")
            print(f"ΔX: {dx:.2f} mm, ΔY: {dy:.2f} mm, ΔZ: {dz:.2f} mm")
            print(f"ΔRoll: {droll:.2f}°, ΔPitch: {dpitch:.2f}°, ΔYaw: {dyaw:.2f}°")
            


    # ---------- Visualization ----------

    tx, ty, tz = pose["tx"], pose["ty"], pose["tz"]
    roll, pitch, yaw = pose["roll"], pose["pitch"], pose["yaw"]

    distance = np.linalg.norm(pose["tvec"])
    x_cam, z_cam = tx, tz
    axis_length = tz / 3

    yaw_rad = -np.deg2rad(yaw)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    R_yaw = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    x_axis_2d = R_yaw @ np.array([axis_length, 0])
    z_axis_2d = R_yaw @ np.array([0, axis_length])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw camera
    triangle_size = 100
    triangle = np.array([[triangle_size, 0],
                         [-triangle_size, 0],
                         [0, -triangle_size / 2]])
    cam_triangle = Polygon(triangle, closed=True, color='red')
    ax.add_patch(cam_triangle)
    ax.text(20, 30, 'Camera', color='r', fontsize=12, verticalalignment='top')
    ax.arrow(0, 0, 0, axis_length * 0.5, color='k', width=3, head_width=10.0, label='Camera View')

    # Object marker
    ax.plot(x_cam, z_cam, 'o', color='gray')
    ax.text(x_cam, z_cam + 20, 'Object', color='gray', fontsize=12)

    # Footprint
    square_size = 177.8
    half_square = square_size / 2
    square_local = np.array([
        [half_square, half_square],
        [-half_square, half_square],
        [-half_square, -half_square],
        [half_square, -half_square]
    ])
    square_rotated = (R_yaw @ square_local.T).T
    square_world = square_rotated + np.array([tx, tz])
    rotated_square = Polygon(square_world, closed=True, edgecolor='gray', facecolor='lightgray', alpha=0.5)
    ax.add_patch(rotated_square)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title("Top-Down View of Imaging Scene in Camera Frame")
    ax.grid(True)
    ax.axis('equal')
    # ax.legend()

    # Pose Summary
    summary_text = (
        f"Distance to object center: {distance:.1f} mm\n"
        f"Object center position in camera frame: X={tx:.1f} mm, Y={ty:.1f} mm, Z={tz:.1f} mm\n"
        # f"Object coordinate system relative to camera frame: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°"
    )
    ax.text(0.5, -0.2, summary_text, ha='center', transform=ax.transAxes, fontsize=11)
    
    
    plt.tight_layout()
    ax.set_autoscale_on(False)
    plt.show()
    
    
    
def draw_translation_annotations_with_scale(img, rvec, tvec, K, dist_coeffs):
    """
    Draws translation annotations and adds scale axes around the image
    to indicate camera frame coordinates:
    +X right from center, +Y down from center.
    """
    img_vis = draw_translation_annotations(img, rvec, tvec, K, dist_coeffs)

    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_vis)
    ax.set_title("Object Pose in Camera Frame")

    # Set axis ticks to camera frame coordinates
    x_ticks = np.linspace(0, h, 9).astype(int)
    y_ticks = np.linspace(0, w, 9).astype(int)

    ax.set_xticks(y_ticks)
    ax.set_yticks(x_ticks)

    ax.set_xticklabels((y_ticks - center_x).astype(int))
    ax.set_yticklabels((x_ticks - center_y).astype(int))

    ax.set_xlabel("Camera X (pixels)")
    ax.set_ylabel("Camera Y (pixels)")

    ax.grid(True, linestyle='--', alpha=0.5)

    tx, ty, tz = tvec.ravel()
    rmat, _ = cv2.Rodrigues(rvec)
    r = R.from_matrix(rmat)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    distance = np.linalg.norm(tvec)

    summary_text = (
        # f"Distance to object center: {distance:.1f} mm\n"
        f"Object center position in camera frame: X={tx:.1f} mm, Y={ty:.1f} mm, Z={tz:.1f} mm\n"
        f"Object coordinate system relative to camera frame: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°"
    )

    ax.text(
        0.5, -0.08, summary_text,
        ha='center', va='top',
        transform=ax.transAxes,
        fontsize=13
    )

    plt.tight_layout()
    plt.show()



def draw_translation_annotations(img, rvec, tvec, K, dist_coeffs):
    """
    Draws:
    - Thick X and Y translation arrows from image center to CAD origin.
    - A dot at the CAD origin.
    - 3D CAD coordinate axes (X: red, Y: green, Z: blue) centered at origin.
    """
    img_vis = img.copy()
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Project CAD origin
    origin_3d = np.array([[0, 0, 0]], dtype=np.float32)
    origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, K, dist_coeffs)
    x_proj, y_proj = origin_2d[0, 0].astype(int)

    # === Y arrow: vertical from image center to projected point ===
    cv2.arrowedLine(
        img_vis,
        center,
        (center[0], y_proj),
        color=(0, 128, 0),
        thickness=24,
        tipLength=0.8
    )
    cv2.putText(
        img_vis,
        f"Y = {tvec[1][0]:.1f} mm",
        (center[0] + 0, (y_proj + center[1]) // 2 + 250),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=4.0,
        color=(0, 128, 0),
        thickness=12,
        lineType=cv2.LINE_AA
    )

    # === X arrow: horizontal from intermediate Y point ===
    intermediate_point = (center[0], y_proj)
    cv2.arrowedLine(
        img_vis,
        intermediate_point,
        (x_proj, y_proj),
        color=(0, 140, 255),
        thickness=24,
        tipLength=0.2
    )
    cv2.putText(
        img_vis,
        f"X = {tvec[0][0]:.1f} mm",
        ((x_proj + center[0]) // 2 + 100, y_proj - 100),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=4.0,
        color=(0, 140, 255),
        thickness=12,
        lineType=cv2.LINE_AA
    )

    # CAD origin
    cv2.circle(img_vis, (x_proj, y_proj), radius=40, color=(0, 0, 0), thickness=-1)

    # === Draw 3D object axes ===
    axis_lengths = [150, 150, 150]
    axes_3d = np.float32([
        [axis_lengths[0], 0, 0],
        [0, axis_lengths[1], 0],
        [0, 0, axis_lengths[2]],
    ]).reshape(-1, 3)
    axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist_coeffs)
    axes_2d = axes_2d.reshape(-1, 2).astype(int)

    cv2.arrowedLine(img_vis, (x_proj, y_proj), tuple(axes_2d[0]), color=(255, 0, 0), thickness=12, tipLength=0.3)  # X
    cv2.arrowedLine(img_vis, (x_proj, y_proj), tuple(axes_2d[1]), color=(0, 255, 0), thickness=12, tipLength=0.3)  # Y
    cv2.arrowedLine(img_vis, (x_proj, y_proj), tuple(axes_2d[2]), color=(0, 0, 255), thickness=12, tipLength=0.3)  # Z

    # Axis labels
    label_offset = np.array([20, 0])
    subscript_offset = np.array([350, 30])

    cv2.putText(img_vis, 'Object', tuple(axes_2d[0] + label_offset), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 0, 0), 12)
    cv2.putText(img_vis, 'Object', tuple(axes_2d[1] + label_offset), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 0), 12)
    cv2.putText(img_vis, 'Object', tuple(axes_2d[2] + label_offset + np.array([-500, 0])), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 255), 12)

    cv2.putText(img_vis, 'x', tuple(axes_2d[0] + label_offset + subscript_offset), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 0, 0), 8)
    cv2.putText(img_vis, 'y', tuple(axes_2d[1] + label_offset + subscript_offset), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 0), 8)
    cv2.putText(img_vis, 'z', tuple(axes_2d[2] + label_offset + np.array([-500, 0]) + subscript_offset), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 255), 8)

    return img_vis






"""
Main Function Summary
---------------------
- Loads a real image and a reference image, along with a JSON file that contains 2D–3D keypoint mappings.
- Estimates the pose (rvec, tvec) of the object using solvePnP based on corner matches.
- Also estimates the reference pose if `compute_ref_pose=True`.
- Explains and visualizes the real and reference poses numerically and graphically:
    - Console output for 6DOF pose
    - Top-down pose plot
    - Annotated camera-frame translation overlay on the image
"""

if __name__ == "__main__":
    
    #==========USER CONFIG=============
    real_img_path = r"Path to new image"
    reference_img_path = r"Path to reference image"
    json_path = r"Path to 2D-3D keypoint mapping .json file"
    

    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    #==================================



    rvec, tvec, _, _, rvec_ref, tvec_ref = estimate_pose(real_img_path, reference_img_path, json_path, K, dist_coeffs, compute_ref_pose=True)
    
    
    if rvec is not None and tvec is not None:
        explain_pose(rvec, tvec, rvec_ref, tvec_ref, Verbose=True)
    else:
        print("Pose estimation/interpretation failed.")
        
        
        
    img_bgr = cv2.imread(real_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    
    draw_translation_annotations_with_scale(img_rgb, rvec, tvec, K, dist_coeffs)

    
    





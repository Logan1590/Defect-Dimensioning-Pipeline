import os
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
import PIL.Image
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import time 
from PIL import Image

from Image_Comparison_v2 import process_image_differences
from visibility_utils_v4 import get_visible_mesh, project_points
from pose_estimation_v11 import estimate_pose


"""
Function Summaries
------------------

load_and_transform_obj(obj_path)
    Loads a .obj mesh file, extracts the relevant geometry (if a scene), applies scaling and translation
    to center and align the mesh.

    Parameters:
        obj_path (str): Path to the .obj file.

    Returns:
        mesh (trimesh.Trimesh): Transformed 3D mesh.

get_visible_submesh(mesh, reference_image_path, rvec, tvec, K, dist_coeffs)
    Extracts the visible subset of the mesh given camera pose and intrinsics by applying visibility culling.

    Parameters:
        mesh (trimesh.Trimesh): Full CAD mesh.
        reference_image_path (str): Path to image used to determine visibility.
        rvec (np.ndarray): Rotation vector.
        tvec (np.ndarray): Translation vector.
        K (np.ndarray): Camera intrinsic matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.

    Returns:
        visible_mesh (trimesh.Trimesh): Visible portion of the mesh.
    
render_model_to_image(mesh, R, t, K, dist_coeffs, image_shape)
    Projects a textured mesh onto the image plane using UV mapping and fills each triangle based on texture sampling.

    Parameters:
        mesh (trimesh.Trimesh): Textured 3D mesh.
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
        K (np.ndarray): Camera intrinsics.
        dist_coeffs (np.ndarray): Distortion coefficients.
        image_shape (tuple): Output image shape (H, W, 3).

    Returns:
        render (np.ndarray): Rendered RGB image.

generate_difference_heatmap(real_img, render_img)
    Creates and displays a 3-panel visualization showing real image, rendered projection, and pixelwise difference heatmap.

    Parameters:
        real_img (np.ndarray): RGB real image.
        render_img (np.ndarray): RGB rendered model projection.

    Returns:
        heatmap (np.ndarray): RGB heatmap image.
        diff (np.ndarray): Grayscale absolute difference image.

load_restriction_mask(fname, restriction_masks_dir)
    Loads an object segmentation mask and subtracts an optional hole mask to define valid regions.

    Parameters:
        fname (str): Filename of the target image.
        restriction_masks_dir (str): Directory containing object and optional "_Hole" masks.

    Returns:
        mask (np.ndarray or None): Boolean array mask of valid region, or None if not found.

render_textured_model(image_path, reference_path, json_keypoints_path, K, dist_coeffs, save_dir, visualize=True)
    Renders a textured mesh using estimated pose, computes a difference heatmap, and optionally saves/visualizes results.

    Parameters:
        image_path (str): Path to input real image.
        reference_path (str): Path to reference image for pose estimation.
        json_keypoints_path (str): Path to keypoint mapping file.
        K (np.ndarray): Camera intrinsics.
        dist_coeffs (np.ndarray): Camera distortion coefficients.
        save_dir (str): Directory to save the rendered image.
        visualize (bool): Whether to show the heatmap and intermediate results.

    Returns:
        render_img (np.ndarray): Rendered RGB projection of the textured model.

get_or_load_visible_mesh(mesh, image_path, R, t, K, dist_coeffs, image_shape, cache_dir)
    Checks if a precomputed visible mesh exists on disk; if not, computes and caches it.

    Parameters:
        mesh (trimesh.Trimesh): Full mesh.
        image_path (str): Image filename (used for cache naming).
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
        K (np.ndarray): Intrinsics matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.
        image_shape (tuple): Shape of the image (H, W).
        cache_dir (str): Directory to store cached visible meshes.

    Returns:
        visible_mesh (trimesh.Trimesh): Visible mesh (with material).
"""







def load_and_transform_obj(obj_path):
    mesh = trimesh.load(obj_path)
    if isinstance(mesh, trimesh.Scene):
        target_name1 = "Material.007"
        target_name2 = "SampleCAD3_Solid3"
        if target_name1 in mesh.geometry:
            mesh = mesh.geometry[target_name1]
        elif target_name2 in mesh.geometry:
            mesh = mesh.geometry[target_name2]
        else:
            raise ValueError("Target geometry not found in .obj scene.")
    mesh.apply_scale(1000.0)
    mesh.apply_scale([1, -1, -1])
    min_bounds = mesh.bounds[0]
    shift_x = -((min_bounds[0] + mesh.bounds[1][0]) / 2)
    shift_z = -((min_bounds[2] + mesh.bounds[1][2]) / 2)
    custom_y_shift = -(min_bounds[1] + 174.1 - 12.7)
    mesh.apply_translation([shift_x, custom_y_shift, shift_z])
    return mesh



def get_visible_submesh(mesh, reference_image_path, rvec, tvec, K, dist_coeffs):
    real_img = cv2.imread(reference_image_path)
    image_shape = real_img.shape[:2]

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    mask = get_visible_mesh(mesh, R, t, K, dist_coeffs, image_shape)
    visible_mesh = mesh.submesh([mask], append=True)

    return visible_mesh



def render_model_to_image(mesh, R, t, K, dist_coeffs, image_shape):
    render = np.zeros(image_shape, dtype=np.uint8)
    texture = np.asarray(mesh.visual.material.image)
    h_tex, w_tex = texture.shape[:2]

    face_uvs = mesh.visual.uv[mesh.faces]

    for face_idx, face in enumerate(mesh.faces):
        tri_3d = mesh.vertices[face]
        tri_uv = face_uvs[face_idx]

        pts_2d = project_points(tri_3d, R, t, K, dist_coeffs)
        pts_2d_int = pts_2d.astype(np.int32)

        # Map UV coordinates to texture pixel coordinates
        tex_coords = (tri_uv * np.array([w_tex, h_tex])).astype(np.int32)
        tex_coords[:, 1] = h_tex - tex_coords[:, 1]  # Flip V-axis

        # Get triangle color by sampling the texture at the centroid of UVs
        uv_center = np.mean(tex_coords, axis=0).astype(int)
        if not (0 <= uv_center[0] < w_tex and 0 <= uv_center[1] < h_tex):
            continue
        sampled_color = texture[uv_center[1], uv_center[0]]

        cv2.fillConvexPoly(render, pts_2d_int, color=sampled_color.tolist())
    return render



def generate_difference_heatmap(real_img, render_img):
    real_gray = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    render_gray = cv2.cvtColor(render_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = cv2.absdiff(real_gray, render_gray)

    # Normalize diff to 0-1
    diff_norm = cv2.normalize(diff, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Use a custom black-to-red colormap (matplotlib)
    red_cmap = mcolors.LinearSegmentedColormap.from_list("black_to_red", [(0, "black"), (1, "red")])
    heatmap = cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap=red_cmap).to_rgba(diff_norm, bytes=True)[..., :3]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(real_img)
    axs[0].set_title("Real Image")
    axs[1].imshow(render_img)
    # axs[1].set_title("Reference Image")
    axs[1].set_title("Projection of Textured Model")
    axs[2].imshow(heatmap)
    axs[2].set_title("Defect Heatmap (|Real - Projection|)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    return heatmap, diff



def load_restriction_mask(fname, restriction_masks_dir):
    """
    Load the restriction mask and subtract a corresponding hole mask if it exists.

    Returns:
        mask (H, W) as bool numpy array, or None if not found
    """
    base_name = os.path.splitext(fname)[0]
    possible_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    object_mask = None
    hole_mask = None

    # Load main restriction mask
    for ext in possible_exts:
        candidate = os.path.join(restriction_masks_dir, base_name + ext)
        if os.path.exists(candidate):
            img = Image.open(candidate).convert("L")
            object_mask = np.array(img) > 0
            break

    # Load hole mask if present
    for ext in possible_exts:
        candidate = os.path.join(restriction_masks_dir, base_name + "_Hole" + ext)
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



def render_textured_model(image_path, reference_path, json_keypoints_path, K, dist_coeffs, save_dir, visualize=True):

    mesh = load_and_transform_obj(obj_path)
    # Ensure UVs exist
    assert mesh.visual.uv is not None, "Mesh has no UV coordinates."

    # Manually load texture
    texture_path = Path(obj_path).with_name("Path to .png texture file")
    mesh.visual.material.image = PIL.Image.open(texture_path)


    rvec, tvec, *_ = estimate_pose(image_path, reference_path, json_keypoints_path, K, dist_coeffs, compute_ref_pose=False)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    real_img = cv2.imread(image_path)
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
    image_shape = real_img.shape
    
    visible_mesh = get_or_load_visible_mesh(mesh, image_path, R, t, K, dist_coeffs, image_shape[:2], visible_mesh_dir)

    render_img = render_model_to_image(visible_mesh, R, t, K, dist_coeffs, image_shape)


    # Optionally visualize
    if visualize:
        generate_difference_heatmap(real_img, render_img)

    # Save rendered image
    image_name = os.path.basename(image_path)
    rendered_path = os.path.join(save_dir, image_name)
    cv2.imwrite(rendered_path, cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR))

    return render_img



def get_or_load_visible_mesh(mesh, image_path, R, t, K, dist_coeffs, image_shape, cache_dir):
    """
    Load a cached visible mesh for the image, or compute and save it.
    """
    fname = os.path.splitext(os.path.basename(image_path))[0]
    cache_path = os.path.join(cache_dir, fname + ".glb")

    if os.path.exists(cache_path):
        visible_mesh = trimesh.load(cache_path, force='mesh')
        return visible_mesh

    mask = get_visible_mesh(mesh, R, t, K, dist_coeffs, image_shape)
    visible_mesh = mesh.submesh([mask], append=True)
    visible_mesh.visual.material.image = mesh.visual.material.image

    
    visible_mesh.export(cache_path)
    return visible_mesh


    


"""
Main Function Summary
---------------------

This script projects a textured CAD model onto real RGB images using estimated 6DOF camera pose,
generates rendered views, compares them with the real image to detect differences, and evaluates segmentation quality.

Steps performed:
1. Load and transform the CAD mesh from .obj format, including texture assignment and UV validation.
2. Estimate the camera pose from real/reference image pairs using keypoint mappings.
3. Compute visible surface mesh based on the camera view and cache results for efficiency.
4. Render the visible textured mesh into a 2D image using UV-based triangle sampling.
5. Compute a difference heatmap between the rendered image and the real photo.
6. Optionally calculate and visualize restricted IoU between predicted and ground truth defect masks,
   constrained within object segmentation masks.
7. Supports batch processing of single images or entire folders.

Inputs and paths (set near top of main):
- `obj_path`: Path to CAD model (.obj)
- `input_path`: Image file or folder to analyze
- `reference_path`: Used for pose estimation
- `json_keypoints_path`: Keypoint mapping file
- `save_dir`, `visible_mesh_dir`: For rendered images and cached meshes
- `defect_mask_dir`, `object_mask_dir`: For restricted IoU evaluation

Output:
- Saves rendered projections
- Displays visualizations for rendering, defect heatmap, and restricted IoU (if enabled)
- Prints timing and segmentation metrics
"""


if __name__ == "__main__":

    start_time = time.perf_counter()
 
    # ============USER CONFIG ================
    rendered_path = r"Path to save texture renderings to"
    obj_path = r"Path to .obj file"
    input_path = r"Path to new image"
    reference_path = r"Path to reference image"
    json_keypoints_path = r"Path to 2D-3D keypoint mapping .json file"
    save_dir = r"Directory to save texture renderings to"
    visible_mesh_dir = r"Path to directory holding pregenerated visible meshes"   # This does not work
    texture_path = "Path to generated .png texture file"

    visualize = True  # Toggle for visualization
    
    defect_mask_dir = r"Path to directory holding defect masks"
    object_mask_dir = r"Path to directory holding object masks"
    # ============================================
    
    
    
    # === Load CAD mesh and texture ===
    mesh = load_and_transform_obj(obj_path)
    assert mesh.visual.uv is not None, "Mesh has no UV coordinates."
    texture_path = Path(obj_path).with_name(texture_path)
    mesh.visual.material.image = PIL.Image.open(texture_path)
    
    
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    
    # === Precompute pose of reference image to get visibility mask ===
    rvec_ref, tvec_ref, *_ = estimate_pose(reference_path, reference_path, json_keypoints_path, K, dist_coeffs, compute_ref_pose=True)
    


    if os.path.isfile(input_path):
        # Single file
        image_files = [input_path]
    elif os.path.isdir(input_path):
        # Directory of images
        image_files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    for image_path in image_files:
        fname = os.path.basename(image_path)
        print(f"Processing {fname}...")

        try:
            render_img = render_textured_model(
                image_path=image_path,
                reference_path=reference_path,
                json_keypoints_path=json_keypoints_path,
                K=K,
                dist_coeffs=dist_coeffs,
                save_dir=save_dir,
                visualize=visualize
            )


            rendered_path = os.path.join(save_dir, fname)
            
            
            # === Save rendered image using original image filename ===
            image_name = os.path.basename(image_path)
            rendered_path = os.path.join(save_dir, image_name)
            cv2.imwrite(rendered_path, cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR))
            
            
            # === Optional visualization ===
            if visualize:
                real_img = cv2.imread(image_path)
                real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
                generate_difference_heatmap(real_img, render_img)
            
            
                diff, binary, mask = process_image_differences(
                    image_path, rendered_path,
                    threshold=70, kernel_size=9, min_blob_area=1500, mode='Open'
                )
                
                
                # === Calculate restricted IoU (only if visualize is enabled and both mask dirs are provided) ===
                if visualize and defect_mask_dir and object_mask_dir:
                    base_name = os.path.splitext(fname)[0]
                    possible_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
                    
                    # Load defect ground truth mask
                    gt_mask_path = None
                    for ext in possible_exts:
                        candidate = os.path.join(defect_mask_dir, base_name + ext)
                        if os.path.exists(candidate):
                            gt_mask_path = candidate
                            break
                    
                    # Load object restriction mask
                    object_mask = load_restriction_mask(fname, object_mask_dir)
                
                    if gt_mask_path and object_mask is not None:
                        true_mask = np.array(Image.open(gt_mask_path).convert("L")) > 0
                        restricted_true = np.logical_and(true_mask, object_mask)
                        restricted_pred = np.logical_and(mask, object_mask)
                        
                        intersection = np.logical_and(restricted_true, restricted_pred)
                        union = np.logical_or(restricted_true, restricted_pred)
                        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                        print(f"{fname}: IoU (within object mask) = {iou:.3f}")
                
                        # Optional visualization
                        vis_overlay = real_img.copy()
                        vis_overlay[np.logical_and(restricted_pred, ~restricted_true)] = [255, 0, 0]  # red = false positives
                        vis_overlay[intersection] = [0, 255, 0]  # green = true positives
                        plt.figure(figsize=(10, 8))
                        plt.imshow(vis_overlay)
                        plt.title(f"{fname} - Restricted IoU = {iou:.3f}")
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                    else:
                        print(f"{fname}: Skipping IoU — missing GT mask or object mask.")


        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

    
    end_time = time.perf_counter()
    print(f"Script execution time: {end_time - start_time:.4f} seconds")

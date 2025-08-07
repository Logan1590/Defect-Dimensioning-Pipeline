import os
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import time

from visibility_utils_v4 import project_points, get_visible_mesh
from pose_estimation_v12 import estimate_pose

angle_histogram = defaultdict(int)



"""
Function Summaries
------------------

save_texture(texture, out_path):
    - Saves the texture image to disk using OpenCV, converting RGB to BGR format.

report_surface_coverage(mesh, textured_faces_mask):
    - Calculates and prints how much surface area of the mesh was covered by the baked texture.

plot_texture_histogram(texture):
    - Plots RGB and grayscale histograms of the texture image.
    - Excludes pixels with near-zero values to improve visualization of meaningful data.

overlay_entire_mesh_on_image(mesh, image, R, t, K, dist_coeffs, title):
    - Projects the entire mesh into a 2D image using camera intrinsics and pose.
    - Overlays green triangle edges for all faces.

overlay_visible_mesh_on_image(mesh, visible_faces_mask, image, R, t, K, dist_coeffs, title):
    - Projects and overlays only the visible faces (after visibility filtering) onto the image in red.

bake_texture(mesh, views, texture_size, dist_coeffs, visualize):
    - Performs UV texture baking by:
        1. Determining visible faces in each view.
        2. Sampling color from real images using triangle center projection.
        3. Accumulating color data per face.
        4. Filling the texture using UV coordinates and averaging overlapping contributions.
    - Optionally visualizes mesh overlays and the texture histogram.
    - Returns the final texture image as a NumPy uint8 array.
"""


def save_texture(texture, out_path):
    cv2.imwrite(out_path, cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))



def report_surface_coverage(mesh, textured_faces_mask):
    areas = mesh.area_faces
    total_area = np.sum(areas)
    textured_area = np.sum(areas[textured_faces_mask])
    percent_textured = 100 * textured_area / total_area
    print(f"Surface area textured: {textured_area:.2f} / {total_area:.2f} ({percent_textured:.2f}%)")



def plot_texture_histogram(texture):
    flat = texture.reshape(-1, 3)
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(12, 6))
    mask = np.all((flat > 1), axis=1)
    filtered_flat = flat[mask]
    zero_one_mask = np.any((flat <= 1), axis=1)
    excluded_count = np.sum(zero_one_mask)
    for i in range(3):
        plt.hist(filtered_flat[:, i], bins=256, range=(0, 255), color=colors[i], alpha=0.6, label=f'{colors[i].capitalize()} channel')
    gray = np.mean(filtered_flat, axis=1)
    plt.hist(gray, bins=256, range=(0, 255), color='gray', alpha=0.4, label='Grayscale mean')
    plt.title("Histogram of Baked Texture")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Pixel count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.annotate(f"{excluded_count:,} pixels ≤ 1 ignored", xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=10, color='gray')
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()



def overlay_entire_mesh_on_image(mesh, image, R, t, K, dist_coeffs, title="Projected Mesh Overlay"):
    img_h, img_w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    ax.set_title(title)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    for face in mesh.faces:
        tri_3d = mesh.vertices[face]
        pts = project_points(tri_3d, R, t, K, dist_coeffs)
        ax.plot(*pts[[0, 1]].T, 'g-', linewidth=0.5)
        ax.plot(*pts[[1, 2]].T, 'g-', linewidth=0.5)
        ax.plot(*pts[[2, 0]].T, 'g-', linewidth=0.5)
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    
    
    
def overlay_visible_mesh_on_image(mesh, visible_faces_mask, image, R, t, K, dist_coeffs, title="Visible Mesh Overlay"):
    img_h, img_w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    ax.set_title(title)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    for i, face in enumerate(mesh.faces):
        if not visible_faces_mask[i]:
            continue
        tri_3d = mesh.vertices[face]
        pts = project_points(tri_3d, R, t, K, dist_coeffs)
        ax.plot(*pts[[0, 1]].T, 'r-', linewidth=0.5)
        ax.plot(*pts[[1, 2]].T, 'r-', linewidth=0.5)
        ax.plot(*pts[[2, 0]].T, 'r-', linewidth=0.5)
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    
    

def bake_texture(mesh, views, texture_size=(1024, 1024), dist_coeffs=None, visualize=True):
    
    bake_start_time = time.time()

    
    uv = mesh.visual.uv
    faces = mesh.faces
    face_uvs = uv[faces]

    texture = np.zeros((texture_size[1], texture_size[0], 3), dtype=np.float32)
    count_map = np.zeros((texture_size[1], texture_size[0]), dtype=np.uint16)
    textured_faces_mask = np.zeros(len(faces), dtype=bool)

    # Precompute per-view visibility masks
    view_visible_masks = []
    for idx, view in enumerate(views):
        
        mask = get_visible_mesh(mesh, view["R"], view["t"], view["K"], dist_coeffs, view["image"].shape[:2], visualize=False)
        view_visible_masks.append(mask)

        
        # Visualize once during this precomputation to avoid recomputation later
        if visualize:
            overlay_entire_mesh_on_image(mesh, view["image"], view["R"], view["t"], view["K"], dist_coeffs, title=f"Full Mesh Overlay - {view.get('name')}")
            overlay_visible_mesh_on_image(mesh, mask, view["image"], view["R"], view["t"], view["K"], dist_coeffs, title=f"Visible Mesh Overlay - {view.get('name')}")

        split_time = time.time()
        print('Time:', split_time - bake_start_time)

    for i, (face, face_uv) in enumerate(zip(faces, face_uvs)):
        tri_3d = mesh.vertices[face]
        tri_center = np.mean(tri_3d, axis=0)
        face_color_accum = np.zeros((3,), dtype=np.float32)
        valid_view_count = 0

        for j, view in enumerate(views):
            if not view_visible_masks[j][i]:
                continue

            img, R, t, K = view["image"], view["R"], view["t"], view["K"]
            tri_3d_cam = (R @ tri_3d.T + t).T

            tri_center_cam = np.mean(tri_3d_cam, axis=0) 
            projected = project_points(np.array([tri_center]), R, t, K, dist_coeffs)[0]

            u, v = int(round(projected[0])), int(round(projected[1]))
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                sampled_color = img[v, u].astype(np.float32) / 255.0
                face_color_accum += sampled_color
                valid_view_count += 1

        if valid_view_count == 0:
            continue

        textured_faces_mask[i] = True
        avg_color = face_color_accum / valid_view_count
        rgb_color = tuple(int(c * 255) for c in avg_color)

        uv_coords = (face_uv * np.array(texture_size)).astype(np.int32)
        uv_coords[:, 1] = texture_size[1] - uv_coords[:, 1]
        uv_coords = uv_coords.reshape((-1, 1, 2))
        
        # Fill a temporary image and count map with this face's RGB and weight
        face_img = np.zeros_like(texture, dtype=np.float32)
        face_mask = np.zeros(count_map.shape, dtype=np.uint16)
        weighted_color = tuple((avg_color * valid_view_count).tolist())  # still in 0–1
        cv2.fillConvexPoly(face_img, uv_coords, color=weighted_color)
        cv2.fillConvexPoly(face_mask, uv_coords, color=valid_view_count)
        
        texture += face_img
        count_map += face_mask

    
    texture /= np.maximum(count_map[..., None], 1)
    texture = np.clip(texture, 0, 1) * 255


    total_pixels = texture_size[0] * texture_size[1]
    textured_pixels = np.count_nonzero(count_map)
    print(f"UV space usage: {textured_pixels:,} / {total_pixels:,} ({(textured_pixels / total_pixels) * 100:.2f}%)")
    
    
    if visualize:
        plot_texture_histogram(texture)

    return (np.clip(texture, 0, 255)).astype(np.uint8)




"""
Main Function Summary
---------------------
- Loads or computes object poses for each image using 2D–3D keypoints and pose estimation.
- Loads a CAD mesh (.obj) and applies custom scale and translation transformations to align it.
- Applies UV texture baking using the aligned CAD mesh and estimated views.
- Saves the resulting texture image and reports texture coverage and script runtime.
"""

def main():
    
    start_time = time.perf_counter()
    
    #==========USER CONFIG=============
    # Define filepaths
    obj_path = r"Path to .obj file"
    images_dir = r"Path to images folder"
    reference_images_dir = r"Path to folder with reference images"
    json_keypoints_path = r"Path to 2D-3D keypoint mapping .json file"
    views_path = r"Path to .pkl files"   # Optional

    
    # Define camera intrinsics
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    #===================================




    if os.path.exists(views_path):
        print("Loading views from cache...")
        with open(views_path, "rb") as f:
            views = pickle.load(f)
    else:
        views = []
        image_exts = [".jpg", ".jpeg", ".png"]
        for fname in sorted(os.listdir(images_dir)):
            print("Analyzing", fname)
            if not any(fname.lower().endswith(ext) for ext in image_exts):
                continue
            real_img_path = os.path.join(images_dir, fname)
            ref_img_path = os.path.join(reference_images_dir, fname)
            try:
                rvec, tvec, *_ = estimate_pose(real_img_path, ref_img_path, json_keypoints_path, K, dist_coeffs)
            except Exception as e:
                print(f"Pose estimation failed for {fname}: {e}")
                continue
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
            img_bgr = cv2.imread(real_img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            views.append({"image": img_rgb, "R": R, "t": t, "K": K, "name": fname})

    mesh = trimesh.load(obj_path)
    print("Available geometries:", mesh.geometry.keys())

    if isinstance(mesh, trimesh.Scene):
        target_name = "Material.007"  
        target_name2 = "SampleCAD3_Solid3"

        if target_name in mesh.geometry:
            mesh = mesh.geometry[target_name]
        elif target_name2 in mesh.geometry:
            mesh = mesh.geometry[target_name2]
        else:
            raise ValueError(f"Object '{target_name}' not found in OBJ file.")


    mesh.apply_scale(1000.0)
    mesh.apply_scale([1, -1, -1])
    min_bounds = mesh.bounds[0]
    max_bounds = mesh.bounds[1]
    center_bounds = (min_bounds + max_bounds) / 2
    shift_x = -center_bounds[0]
    shift_z = -center_bounds[2]
    custom_y_shift = -(min_bounds[1] + 174.1 - 12.7)
    shift_vector = np.array([shift_x, custom_y_shift, shift_z])
    mesh.apply_translation(shift_vector)

    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        print("Mesh does not have UV coordinates. Cannot proceed with baking.")
        return


    texture = bake_texture(mesh, views, texture_size=(1024, 1024), dist_coeffs=dist_coeffs, visualize=False)
    texture_path = obj_path.replace(".obj", "_baked_texture_v2.png")
    save_texture(texture, texture_path)
    print(f"Texture saved to {texture_path}")


    end_time = time.perf_counter()
    print(f"Script execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()

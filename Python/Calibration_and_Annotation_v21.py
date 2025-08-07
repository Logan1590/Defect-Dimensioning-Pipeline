import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import trimesh
from torchvision.utils import draw_segmentation_masks
from trimesh.ray import ray_pyembree
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import time

from pose_estimation_v13 import estimate_pose
from Pose_Interpretation_v13 import explain_pose, draw_translation_annotations_with_scale
from visibility_utils_v4 import get_visible_mesh, project_points
from Inference_v4_5 import run_feature_detection
from Image_Comparison_v2 import process_image_differences, overlay_mask_on_image
from Model_Based_Defect_Segmentation_v5_5 import generate_difference_heatmap




"""
Function Summaries
------------------

compute_length_width_from_mask(mask_bool):
    - Computes the principal axes of a binary mask using PCA.
    - Projects pixel coordinates onto length and width axes.
    - Returns the physical length, width, mask centroid, axis directions, and coordinate list.

overlay_projected_cad_edges_on_image(image, cad_mesh, rvec, tvec, K, dist_coeffs):
    - Projects CAD mesh edges into the image using pose and camera intrinsics.
    - Draws all projected edges as magenta lines on the image.
    - Returns the annotated image with CAD overlay.

create_cad_projection_mask(cad_mesh, rvec, tvec, K, dist_coeffs, image_shape):
    - Projects all mesh vertices and fills each triangle as a convex polygon.
    - Creates a binary mask indicating the area covered by the CAD model in the image.
    - Returns a boolean NumPy array representing the projection mask.

measure_physical_dimensions(mask_bool, rvec, tvec, K, dist_coeffs, cad_mesh):
    - Approximates the defect region using a polygon or fallback bounding box.
    - Identifies the longest and shortest sides, backprojects their endpoints to 3D.
    - Computes physical lengths using CAD mesh geometry.
    - Returns the 2D points of the measured sides and their physical lengths.

draw_dimension_line(ax, pt1, pt2, color):
    - Draws a main line and perpendicular endcaps between two 2D points on a matplotlib axis.
    - Used for dimension annotation.

refine_mesh_near_defects(cad_mesh, defect_mask, rvec, tvec, K, dist_coeffs, refinement_iters, margin_px):
    - Projects CAD mesh vertices to image space and identifies faces near the defect mask.
    - Subdivides mesh faces within a dilated margin around defects to increase resolution.
    - Merges refined and unrefined mesh parts and returns the updated mesh.

draw_all_polygon_dimensions(ax, approx_pts, cad_mesh, rvec, tvec, K, dist_coeffs, label_origin):
    - Iterates over each polygon edge (up to 4), draws it, and labels it (L1–L4).
    - Computes real-world length of each side by backprojecting 2D endpoints to CAD model.
    - Displays the measured lengths as text annotations on the image.
"""





def compute_length_width_from_mask(mask_bool):
    coords = np.column_stack(np.where(mask_bool))
    if len(coords) < 2:
        return 0, 0, (0, 0), np.eye(2), coords
    pca = PCA(n_components=2)
    pca.fit(coords)
    length_dir = pca.components_[0]
    length_dir /= np.linalg.norm(length_dir)
    width_dir = np.array([-length_dir[1], length_dir[0]])  # force perpendicular
    centered_coords = coords - pca.mean_
    length_proj = np.dot(centered_coords, length_dir)
    width_proj = np.dot(centered_coords, width_dir)
    length = length_proj.max() - length_proj.min()
    width = width_proj.max() - width_proj.min()
    center = pca.mean_
    axes = np.vstack([length_dir, width_dir])
    return length, width, center, axes, coords



def overlay_projected_cad_edges_on_image(image, cad_mesh, rvec, tvec, K, dist_coeffs):
    img_overlay = image.copy()
    if not hasattr(cad_mesh, "edges_unique") or cad_mesh.edges_unique is None:
        cad_mesh.process(validate=True)
    edges = cad_mesh.edges_unique
    vertices = cad_mesh.vertices.copy()
    points_3d_start = vertices[edges[:, 0]]
    points_3d_end = vertices[edges[:, 1]]
    projected_start = project_points(points_3d_start, rvec, tvec, K, dist_coeffs)
    projected_end = project_points(points_3d_end, rvec, tvec, K, dist_coeffs)
    for pt1, pt2 in zip(projected_start, projected_end):
        pt1 = tuple(np.round(pt1).astype(int))
        pt2 = tuple(np.round(pt2).astype(int))
        cv2.line(img_overlay, pt1, pt2, color=(255, 0, 255), thickness=5)
    return img_overlay



def create_cad_projection_mask(cad_mesh, rvec, tvec, K, dist_coeffs, image_shape):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    vertices = cad_mesh.vertices
    faces = cad_mesh.faces
    projected_pts = project_points(vertices, rvec, tvec, K, dist_coeffs)
    projected_pts = projected_pts.astype(np.int32)
    for face in faces:
        pts = projected_pts[face]
        pts = np.clip(pts, [0, 0], [width - 1, height - 1])
        cv2.fillConvexPoly(mask, pts, 1)
    return mask.astype(bool)



def measure_physical_dimensions(mask_bool, rvec, tvec, K, dist_coeffs, cad_mesh):
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, (0.0, 0.0)

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_pts = approx[:, 0]  # shape (N, 2)

    if len(approx_pts) < 3:
        # Fallback to longest line segment approximation
        print("Polygon too simple, falling back to line approximation.")

        # Compute bounding box and use its ends as length endpoints
        x, y, w, h = cv2.boundingRect(contour)
        pt1 = (x, y)
        pt2 = (x + w, y + h)

        # Back-project points to CAD mesh
        projected_2d = project_points(cad_mesh.vertices, rvec, tvec, K, dist_coeffs)
        tree = cKDTree(projected_2d)
        _, idx1 = tree.query(pt1)
        _, idx2 = tree.query(pt2)
        v3d = cad_mesh.vertices
        length = np.linalg.norm(v3d[idx1] - v3d[idx2])

        return (pt1, pt2, pt2, pt2), (length, 0.0)  # Duplicate pt2 for width placeholder

    # Proceed with polygon-based 2-side measurement
    def order_clockwise(pts):
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return pts[np.argsort(angles)]

    ordered = order_clockwise(approx_pts)
    num_pts = len(ordered)
    sides = [np.linalg.norm(ordered[i] - ordered[(i + 1) % num_pts]) for i in range(num_pts)]

    # Find longest and shortest (not necessarily adjacent)
    longest_idx = np.argmax(sides)
    shortest_idx = np.argmin(sides)

    idx1a, idx1b = longest_idx, (longest_idx + 1) % num_pts
    idx2a, idx2b = shortest_idx, (shortest_idx + 1) % num_pts

    pt1a, pt1b = ordered[idx1a], ordered[idx1b]
    pt2a, pt2b = ordered[idx2a], ordered[idx2b]

    # Back-project and measure
    projected_2d = project_points(cad_mesh.vertices, rvec, tvec, K, dist_coeffs)
    tree = cKDTree(projected_2d)
    _, i1a = tree.query(pt1a)
    _, i1b = tree.query(pt1b)
    _, i2a = tree.query(pt2a)
    _, i2b = tree.query(pt2b)
    v3d = cad_mesh.vertices
    len1 = np.linalg.norm(v3d[i1a] - v3d[i1b])
    len2 = np.linalg.norm(v3d[i2a] - v3d[i2b])
    

    return (pt1a, pt1b, pt2a, pt2b), (len1, len2)



def draw_dimension_line(ax, pt1, pt2, color):
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
    perp = np.array([pt2[1] - pt1[1], -(pt2[0] - pt1[0])], dtype=np.float64)
    norm = np.linalg.norm(perp)
    if norm == 0:
        return
    perp = perp / norm * 8
    for pt in [pt1, pt2]:
        ax.plot([pt[0] - perp[0], pt[0] + perp[0]], [pt[1] - perp[1], pt[1] + perp[1]], color=color, linewidth=1)
        
        
        
def refine_mesh_near_defects(cad_mesh, defect_mask, rvec, tvec, K, dist_coeffs, refinement_iters=1, margin_px=300):
    """
    Subdivides mesh faces whose projected vertices fall within an expanded defect mask.
    
    Parameters:
    - cad_mesh: trimesh.Trimesh
    - defect_mask: 2D boolean numpy array
    - rvec, tvec, K, dist_coeffs: Camera parameters
    - refinement_iters: Number of subdivision iterations for selected faces
    - margin_px: Number of pixels to expand around the defect mask

    Returns:
    - refined_mesh: trimesh.Trimesh
    """
    from trimesh import Trimesh

    # Expand the defect mask slightly
    kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)
    expanded_mask = cv2.dilate(defect_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    # Project all vertices
    verts_2d = project_points(cad_mesh.vertices, rvec, tvec, K, dist_coeffs)
    verts_2d = verts_2d.astype(int)

    h, w = defect_mask.shape
    verts_in_mask = np.zeros(len(cad_mesh.vertices), dtype=bool)
    valid = (verts_2d[:, 0] >= 0) & (verts_2d[:, 0] < w) & (verts_2d[:, 1] >= 0) & (verts_2d[:, 1] < h)
    verts_2d_clipped = verts_2d[valid]
    in_mask = expanded_mask[verts_2d_clipped[:, 1], verts_2d_clipped[:, 0]]
    verts_in_mask[np.where(valid)[0][in_mask]] = True

    # Select faces that have >=2 vertices in mask
    face_mask = np.sum(verts_in_mask[cad_mesh.faces], axis=1) >= 2

    # Subdivide selected faces
    if not np.any(face_mask):
        print("No mesh faces found near defect to refine.")
        return cad_mesh

    submesh = cad_mesh.submesh([face_mask], append=True)
    for _ in range(refinement_iters):
        submesh = submesh.subdivide()

    # Keep unrefined faces outside defect regions
    rest_mesh = cad_mesh.submesh([~face_mask], append=True)

    # Combine refined and unrefined parts
    refined_mesh = trimesh.util.concatenate([rest_mesh, submesh])
    print(f"Local mesh refinement at defect regions yields {len(submesh.faces)} faces")

    return refined_mesh

    

def draw_all_polygon_dimensions(ax, approx_pts, cad_mesh, rvec, tvec, K, dist_coeffs, label_origin):
    """
    Draws dimensions for all polygon sides, up to 4, and annotates them as L1–L4.
    """
    from visibility_utils_v4 import project_points
    from scipy.spatial import cKDTree

    if len(approx_pts) < 2:
        return

    num_pts = len(approx_pts)
    colors = ['orange', 'cyan', 'lime', 'magenta']
    projected_2d = project_points(cad_mesh.vertices, rvec, tvec, K, dist_coeffs)
    tree = cKDTree(projected_2d)
    label_x, label_y = label_origin

    for i in range(min(4, num_pts)):
        pt1 = approx_pts[i]
        pt2 = approx_pts[(i + 1) % num_pts]

        # Draw the dimension line
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=colors[i], linewidth=3)
        perp = np.array([pt2[1] - pt1[1], -(pt2[0] - pt1[0])], dtype=float)
        perp /= (np.linalg.norm(perp) + 1e-8)
        perp *= 8
        for pt in [pt1, pt2]:
            ax.plot([pt[0] - perp[0], pt[0] + perp[0]], [pt[1] - perp[1], pt[1] + perp[1]], color=colors[i], linewidth=1)

        # Backproject to 3D
        _, idx1 = tree.query(pt1)
        _, idx2 = tree.query(pt2)
        v3d = cad_mesh.vertices
        length = np.linalg.norm(v3d[idx1] - v3d[idx2])

        # Annotate
        ax.text(label_x, label_y + i * 100, f"L{i+1}: {length:.1f} mm", color=colors[i], fontsize=12,
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))

    
    
    
    
    
    
"""
Main Function Summary
---------------------
- Loads and centers a CAD mesh and subdivides it for initial refinement.
- Iterates through real images and performs the following steps:
    - **Segmentation:** Applies a Mask R-CNN-based segmentation model to identify defect regions.
    - **Pose Estimation:** Estimates the 6DOF pose of the object using keypoint correspondences.
    - **Visibility Filtering:** Filters the CAD mesh to retain only the faces visible from the camera view.
    - **Mesh Refinement:** Further refines mesh resolution in areas near defect regions.
    - **Projection Visualization:** Overlays the visible CAD mesh onto the real image.
    - **Dimensioning:** Measures physical defect dimensions by polygon approximation and CAD backprojection.
    - **Annotation:** Draws polygons and labels length measurements (L1–L4) for up to 3 largest defect regions.
- Displays visualizations and prints timing for each stage of the pipeline.
"""

def main():
    
    start_time = time.perf_counter()

    #===================USER CONFIG=========================
    real_images_dir = r"Path to new image"
    reference_images_dir = r"Path to reference image"
    json_keypoints_path = r"Path to 2D-3D keypoint mapping .json file"
    #=======================================================


    
    # Load CAD
    cad_path = r"Path to object .stl file"
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    cad_mesh = trimesh.load(cad_path)
    center_bounds = (cad_mesh.bounds[0] + cad_mesh.bounds[1]) / 2
    shift_x = -center_bounds[0]
    shift_z = -center_bounds[2]
    custom_y_shift = -(cad_mesh.bounds[0][1] + 174.1 - 12.7)    # Adjust the .stl to align center of bottom with origin
    cad_mesh.apply_translation(np.array([shift_x, custom_y_shift, shift_z]))
    cad_mesh = cad_mesh.subdivide().subdivide().subdivide().subdivide()
    print(f"Number of triangles in CAD mesh: {len(cad_mesh.faces)}")

    
    if isinstance(cad_mesh, trimesh.Scene):
        cad_mesh = cad_mesh.dump(concatenate=True)
        

    
    
    for fname in sorted(os.listdir(real_images_dir)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        print("Analyzing", fname)
        real_img_path = os.path.join(real_images_dir, fname)
        reference_img_path = os.path.join(reference_images_dir, fname)
        
        
        
        # # Segmentation Method 1: ML Model
        segmentation_start_time = time.perf_counter()
        results = run_feature_detection(
            model_path=r"Path to trained defect segmentation ML model (.pth file)",
            images_dir=real_img_path,
            masks_dir=None,
            threshold=0.8,
            visualize=False
        )
        

        if fname not in results:
            continue
        defect_mask, _ = results[fname]
        if defect_mask.sum() == 0:
            continue
        
        defect_mask = defect_mask.astype(np.uint8)

        # Remove small blobs using connected components
        min_blob_area = 300

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask)
        mask = np.zeros_like(defect_mask)
        for i in range(1, num_labels):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                mask[labels == i] = 255
        defect_mask = mask
  
        segmentation_end_time = time.perf_counter()
        print(f"Segmentation time: {segmentation_end_time - segmentation_start_time:.1f} seconds")


        # Visualization
        vis_image = results[fname][1].permute(1, 2, 0).numpy()
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_image)
        plt.title("Segmentation Result")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        

        
        # # # Segmentation Method 2: Perspective Correction + Image Comparison + Defect Processing
        # # # Load image of object after correcting for any difference in perspective from reference image. 
        # segmentation_start_time = time.perf_counter()
        # nondefective_img_path = r'Path to image from corrected perspective'
        # diff, binary, defect_mask = process_image_differences(nondefective_img_path, real_img_path, threshold=70, kernel_size=7, min_blob_area=300, mode='None')
        # segmentation_end_time = time.perf_counter()
        # print(f"Segmentation time: {segmentation_end_time - segmentation_start_time:.1f} seconds")
        # defective_image = cv2.imread(real_img_path)  # BGR
        # real_img = cv2.imread(real_img_path)
        # nondefective_img = cv2.imread(nondefective_img_path)
        # heatmap, diff = generate_difference_heatmap(real_img, nondefective_img)
        # overlay_mask_on_image(defective_image, defect_mask)



        
        
        
        # # # Segmentation Method 3: Textured Model Projection + Image Comparison + Defect Processing
        # # # Load image of 2D rendering (projection of textured model with object pose). Currently, there is no callable function to project a texture on demand
        # segmentation_start_time = time.perf_counter()
        # rendered_path = r"Path to 2D rendering"
        # render_img = cv2.imread(rendered_path)
        # real_img = cv2.imread(real_img_path)
        # real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        # diff, binary, defect_mask = process_image_differences(real_img_path, rendered_path, threshold=70, kernel_size=7, min_blob_area=300, mode='None')
        # segmentation_end_time = time.perf_counter()
        # print(f"Segmentation time: {segmentation_end_time - segmentation_start_time:.1f} seconds")
        # heatmap, diff = generate_difference_heatmap(real_img, render_img)
        # defective_image = cv2.imread(real_img_path)  # BGR
        # overlay_mask_on_image(defective_image, defect_mask)


        
        
        
        # # # Segmentation Method 4: Segmentation by hand with LabelMe
        # segmentation_start_time = time.perf_counter()
        # defect_mask_path = r'Path to defect mask'
        # defect_mask = cv2.imread(defect_mask_path, cv2.IMREAD_GRAYSCALE)
        # defect_mask = (defect_mask > 0).astype(np.uint8)
        # segmentation_end_time = time.perf_counter()
        # print(f"Segmentation time: {segmentation_end_time - segmentation_start_time:.1f} seconds")
        # defective_image = cv2.imread(real_img_path)  # BGR
        # overlay_mask_on_image(defective_image, defect_mask)
        
        
        
        
        
        
        
        
        # Estimate pose
        try:
            pose_estimation_start_time = time.perf_counter()
            rvec, tvec, *_ = estimate_pose(real_img_path, reference_img_path, json_keypoints_path, K, dist_coeffs)
            pose_estimation_end_time = time.perf_counter()
            print(f"Pose estimation time: {pose_estimation_end_time - pose_estimation_start_time:.1f} seconds")
        except Exception as e:
            print(f"Pose estimation failed for {fname}: {e}")
            continue
        
        try:
            explain_pose(rvec,tvec, Verbose=False)
            img_bgr = cv2.imread(real_img_path)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            draw_translation_annotations_with_scale(img_rgb, rvec, tvec, K, dist_coeffs)

        except Exception as e:
            print(f'Pose interpretation failed for {fname}: {e}')
        
        
        
        
        
        
        # Find vislble portion of mesh
        mesh_visibility_start_time = time.perf_counter()

        img_bgr = cv2.imread(real_img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        visible_faces_mask = get_visible_mesh(cad_mesh, R, t, K, dist_coeffs, img_rgb.shape[:2])
        cad_mesh_visible = cad_mesh.submesh([visible_faces_mask], append=True)
        
        mesh_visibility_end_time = time.perf_counter()
        print(f"Mesh visibility time: {mesh_visibility_end_time -  mesh_visibility_start_time:.1f} seconds")

        
        
        
        
        
        # Refine mesh
        mesh_refinement_start_time = time.perf_counter()
        
        cad_mesh_visible = refine_mesh_near_defects(
                                                    cad_mesh_visible, defect_mask, rvec, tvec, K, dist_coeffs,
                                                    refinement_iters=5, margin_px=100
                                                    )
        
        mesh_refinement_end_time = time.perf_counter()
        print(f"Mesh refinement time: {mesh_refinement_end_time - mesh_refinement_start_time:.1f} seconds")
        
        
        
        # Visualize mesh projection
        cad_overlay_img = overlay_projected_cad_edges_on_image(img_rgb.copy(), cad_mesh_visible, rvec, tvec, K, dist_coeffs)
        cad_mask = create_cad_projection_mask(cad_mesh_visible, rvec, tvec, K, dist_coeffs, img_rgb.shape)
        fig, axs = plt.subplots(1, 2, figsize=(24, 12))
        axs[0].imshow(img_rgb)
        axs[0].set_title(f"Defect Annotations for {fname}")
        axs[0].axis('off')
        axs[1].imshow(cad_overlay_img)
        axs[1].set_title(f"{fname} - CAD Projection")
        axs[1].axis('off')
        
        
        
        # Annotate largest 3 defects
        dimensioning_start_time = time.perf_counter()
        
        num_labels, label_map = cv2.connectedComponents(defect_mask.astype(np.uint8))
        region_info = []
        
        for label in range(1, num_labels):
            mask_bool = label_map == label
            intersected_mask = mask_bool & cad_mask
        
            if intersected_mask.sum() < 10:
                continue
        
            dim_result, (phys_length, phys_width) = measure_physical_dimensions(
                intersected_mask, rvec, tvec, K, dist_coeffs, cad_mesh_visible)
            if dim_result is None:
                print('Skipped a region')
                continue
        
            # Polygon approximation and multi-side dimensioning
            contours, _ = cv2.findContours(intersected_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    print("Too few points, skipping")
                    continue
        
                approx_pts = approx[:, 0]
                poly_pts = np.vstack([approx_pts, approx_pts[0]])  # Close loop
                axs[0].plot(poly_pts[:, 0], poly_pts[:, 1], color='red', linewidth=2)
        
                # Measure all sides
                side_lengths = [np.linalg.norm(approx_pts[i] - approx_pts[(i + 1) % len(approx_pts)]) for i in range(len(approx_pts))]
                total_length = sum(side_lengths)
        
                ys, xs = np.where(intersected_mask)
                max_x = np.max(xs)
                min_y = np.min(ys)
                label_origin = (max_x + 50, min_y - 20)
        
                region_info.append({
                    "total_length": total_length,
                    "approx_pts": approx_pts,
                    "label_origin": label_origin
                })
        
        
        # Sort and annotate top 3 by total side length
        region_info_sorted = sorted(region_info, key=lambda x: x["total_length"], reverse=True)
        for i, region in enumerate(region_info_sorted[:3]):
            draw_all_polygon_dimensions(axs[0], region["approx_pts"], cad_mesh_visible, rvec, tvec, K, dist_coeffs, region["label_origin"])
        
        plt.tight_layout()
        plt.show()
        
        dimensioning_end_time = time.perf_counter()
        print(f"Dimensioning time: {dimensioning_end_time - dimensioning_start_time:.1f} seconds")
        print('Defect dimensioning successful.')

           
        
    end_time = time.perf_counter()
    print(f"Script execution time: {end_time - start_time:.1f} seconds")


if __name__ == "__main__":
    main()


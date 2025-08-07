import numpy as np
import cv2
import trimesh
from matplotlib import pyplot as plt
from collections import defaultdict

angle_histogram = defaultdict(int)



"""
---------------------
- This script defines utility functions for determining visible triangles on a 3D mesh from a camera view.
- It is intended for use in mesh projection, texturing, or occlusion-aware analysis workflows.
- When imported as a module, none of the functions execute automatically.
---------------------



Function Summaries
------------------

project_points(points_3d, R, t, K, dist_coeffs):
    - Projects 3D points into 2D image space using a given rotation matrix, translation vector, and camera intrinsics.
    - Returns the resulting 2D coordinates as a NumPy array of shape (N, 2).

backface_culling(tri_3d_cam, tolerance_deg=89):
    - Computes the normal of a triangle and checks whether it is facing toward the camera within a tolerance angle.
    - Returns True if the triangle is front-facing; also updates a global histogram of triangle angles.

rasterize_depth_map(mesh, R, t, K, dist_coeffs, image_shape, visualize=True):
    - Generates a depth map by rasterizing visible triangles of the 3D mesh.
    - Applies backface culling and triangle projection to fill in mean triangle depth per pixel.
    - Optionally visualizes the depth map with masked invalid regions.
    - Returns a (H, W) float array representing per-pixel depth in camera space.

is_triangle_visible(tri_center_cam, image_shape, projected_2d):
    - Checks whether a triangle’s center lies within the image bounds and is in front of the camera.
    - Returns True if visible, False otherwise.

get_visible_mesh(mesh, R, t, K, dist_coeffs, image_shape, visualize=True):
    - Determines which triangles in the mesh are visible from a given camera pose and intrinsics.
    - Uses backface culling, triangle center projection, and depth comparison with a precomputed depth map.
    - Optionally visualizes the depth map and returns a boolean mask indicating visible faces.

"""




def project_points(points_3d, R, t, K, dist_coeffs):
    rvec, _ = cv2.Rodrigues(R)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, t, K, dist_coeffs)
    return points_2d.reshape(-1, 2)



def backface_culling(tri_3d_cam, tolerance_deg=89):
    normal = np.cross(tri_3d_cam[1] - tri_3d_cam[0], tri_3d_cam[2] - tri_3d_cam[0])
    if np.linalg.norm(normal) == 0:
        return False
    normal /= np.linalg.norm(normal)
    center = np.mean(tri_3d_cam, axis=0)
    view_dir = -center / (np.linalg.norm(center) + 1e-8)
    cos_angle = np.clip(np.dot(normal, view_dir), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    rounded_angle = int(round(angle_deg))
    angle_histogram[rounded_angle] += 1
    return rounded_angle <= tolerance_deg



def rasterize_depth_map(mesh, R, t, K, dist_coeffs, image_shape, visualize=True):
    height, width = image_shape
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    for face in mesh.faces:
        tri_3d = mesh.vertices[face]
        tri_3d_cam = (R @ tri_3d.T + t).T
        if not backface_culling(tri_3d_cam):
            continue
        tri_2d = project_points(tri_3d, R, t, K, dist_coeffs)
        if np.any(np.isnan(tri_2d)):
            continue
        tri_2d_int = np.round(tri_2d).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tri_2d_int, 1)
        zs = tri_3d_cam[:, 2]
        z_mean = np.mean(zs)
        update_mask = (mask == 1) & (z_mean < depth_map)
        depth_map[update_mask] = z_mean


    if visualize:
        # Create a masked array for invalid depths (== 0 or inf)
        masked_depth = np.ma.masked_where(~np.isfinite(depth_map) | (depth_map == 0), depth_map)
        
        # Create reversed inferno colormap and set background color
        cmap = plt.get_cmap('viridis_r').copy()
        cmap.set_bad(color='black')  # or 'black', '#888888', (0.5, 0.5, 0.5)
        
        # Only use valid depths for colorbar limits
        valid_depths = masked_depth.compressed()
        vmin = np.min(valid_depths)
        vmax = np.max(valid_depths)
        
        # Plot with masked regions
        plt.imshow(masked_depth, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Depth (mm)')
        plt.title("Depth Map")
        plt.tight_layout()
        plt.show()


    return depth_map



def is_triangle_visible(tri_center_cam, image_shape, projected_2d):
    h, w = image_shape
    u, v = int(round(projected_2d[0])), int(round(projected_2d[1]))
    return 0 <= u < w and 0 <= v < h and tri_center_cam[2] > 0



def get_visible_mesh(mesh, R, t, K, dist_coeffs, image_shape, visualize=True):
    depth_map = rasterize_depth_map(mesh, R, t, K, dist_coeffs, image_shape, visualize=visualize)
    visible_faces_mask = np.zeros(len(mesh.faces), dtype=bool)
    for i, face in enumerate(mesh.faces):
        tri_3d = mesh.vertices[face]
        tri_cam = (R @ tri_3d.T + t).T
        if not backface_culling(tri_cam):
            continue
        tri_center = np.mean(tri_3d, axis=0)  # needed for projection
        tri_center_cam = np.mean(tri_cam, axis=0)  # use tri_cam instead of reprojecting
        projected = project_points(np.array([tri_center]), R, t, K, dist_coeffs)[0]
        if not is_triangle_visible(tri_center_cam, image_shape, projected):
            continue
        u, v = int(round(projected[0])), int(round(projected[1]))
        if 0 <= v < image_shape[0] and 0 <= u < image_shape[1]:
            z = tri_center_cam[2]
            if abs(depth_map[v, u] - z) < 1e-1:
                visible_faces_mask[i] = True

    print('Mesh visibility check successful.')
    return visible_faces_mask

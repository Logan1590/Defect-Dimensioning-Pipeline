import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks





"""
Function Summaries
------------------

process_image_differences(nondefective_image_path, defective_image_path, threshold, kernel_size, min_blob_area, mode):
    - Loads two grayscale images (nondefective and defective).
    - Computes their absolute difference and thresholds it to highlight changes.
    - Applies optional morphological operations (e.g., 'Open', 'Erode') to clean up noise.
    - Filters out small blobs below `min_blob_area` using connected component analysis.
    - Plots four stages: raw difference, binary threshold, morphed mask, and final filtered mask.
    - Returns the grayscale difference image, the thresholded binary mask, and the final defect mask.

overlay_mask_on_image(image_bgr, binary_mask, alpha, show):
    - Overlays a red mask (binary_mask) on top of an input BGR image using transparency alpha.
    - Converts the image to RGB and tensor format, applies the mask with `draw_segmentation_masks`, and converts back to NumPy.
    - Optionally visualizes the overlay using matplotlib.
    - Returns the final overlay image in RGB format as a NumPy array.

"""




def process_image_differences(nondefective_image_path, defective_image_path, threshold=30, kernel_size=5, min_blob_area=100, mode='Open'):
    '''
    Computes and visualizes the pixel-wise differences between a nondefective and defective image.

    Parameters:
        nondefective_image_path (str): Path to the nondefective grayscale image.
        defective_image_path (str): Path to the defective grayscale image.
        threshold (int): Pixel intensity difference threshold for binary segmentation.
        kernel_size (int): Size of the morphological kernel for cleanup.
        min_blob_area (int): Minimum area for blobs to be kept.
        mode (str): Morphological operation mode ('Open', 'Erode', 'None').
    
    Returns:
        diff (np.ndarray): Grayscale difference image.
        binary (np.ndarray): Binary thresholded difference.
        mask (np.ndarray): Final filtered defect mask (binary).
    '''
    
    
    # Load and convert images to grayscale
    img1 = cv2.imread(nondefective_image_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(defective_image_path, cv2.IMREAD_GRAYSCALE)

    
    # Ensure the images are the same size
    if img1.shape != img2.shape:
        print('Image1 shape:', img1.shape)
        print('Image2 shape:', img2.shape)
        raise ValueError("Images must be the same size")
    
    # Compute absolute difference
    diff = cv2.absdiff(img1, img2)
    
    # Apply binary threshold
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    

            
            
    # Morphology
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # e.g. k=3, 5, or 7
    
    if mode=='Erode':
        mask = cv2.erode(binary, kernel, iterations=1)
    elif mode=='Open':
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif mode=='None':
        mask = binary
    kernel_mask = mask
    
    
    
    # Remove small blobs using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
            mask[labels == i] = 255

        
    
    # Plot all results in one figure
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(diff, cmap='gray')
    axes[0].set_title('Difference (Grayscale)')
    axes[0].axis('off')

    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title(f'Thresholded (>{threshold})')
    axes[1].axis('off')

    axes[2].imshow(kernel_mask, cmap='gray')
    axes[2].set_title(f'Morphed ({kernel_size}x{kernel_size} kernel)')
    axes[2].axis('off')
    
    axes[3].imshow(mask, cmap='gray')
    axes[3].set_title(f'Filtered (>{min_blob_area}px)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    return diff, binary, mask



def overlay_mask_on_image(image_bgr, binary_mask, alpha=0.5, show=True):
    """
    Overlays a binary mask in red on a BGR image and returns the overlay as a NumPy RGB image.

    Parameters:
        image_bgr (np.ndarray): Input image in BGR format (as from cv2.imread).
        binary_mask (np.ndarray): Binary mask (same height and width), values 0 or 255 or boolean.
        alpha (float): Transparency level for overlay.
        show (bool): Whether to plot the overlay with matplotlib.

    Returns:
        overlay_image_rgb (np.ndarray): RGB image with red mask overlaid.
    """
    
    
    # Convert BGR image (OpenCV) to RGB tensor (C, H, W)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb)  # [C,H,W], float32 [0,1]

    # Ensure binary mask is boolean and shape matches
    mask_tensor = torch.tensor(binary_mask > 0, dtype=torch.bool).unsqueeze(0)  # [1,H,W]

    # Draw red overlay
    overlay_tensor = draw_segmentation_masks((image_tensor * 255).byte(), mask_tensor, alpha=alpha, colors="red")

    # Convert back to NumPy RGB image
    overlay_image = overlay_tensor.permute(1, 2, 0).cpu().numpy()

    if show:
        plt.figure(figsize=(8, 6))
        plt.imshow(overlay_image)
        plt.title("Segmentation Result")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return overlay_image





"""
Main Function Summary
---------------------
- Loads a reference and a test image.
- Calls `process_image_differences` to compute a defect mask by comparing pixel intensities.
- Calls `overlay_mask_on_image` to visualize the resulting binary defect mask overlaid on the test image.
"""

if __name__ == "__main__":

    #==============USER CONFIG===========
    # Define filepaths    
    ref_image_path = r"Path to reference image"
    test_image_path = r"Path to new image"
    #====================================
    

    diff, binary, defect_mask = process_image_differences(test_image_path, ref_image_path, threshold=70, kernel_size=5, min_blob_area=300, mode='None')
    
    test_image = cv2.imread(test_image_path)  # BGR
    overlay_mask_on_image(test_image, defect_mask)
    
    

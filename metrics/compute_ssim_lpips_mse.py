import os
import cv2
import numpy as np
import sys
import argparse

try:
    from skimage.metrics import structural_similarity as ssim
    import skimage
except ImportError:
    print("Error: Missing required library. Please install scikit-image:")
    print("pip install scikit-image opencv-python numpy")
    sys.exit(1)

try:
    import torch
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: LPIPS library not installed. LPIPS metric will not be available.")
    print("To use LPIPS, install: pip install lpips torch torchvision")
    LPIPS_AVAILABLE = False


def composite_on_background(img, background_color=(255, 255, 255)):
    """Composite RGBA image onto specified background
    
    Args:
        img: RGBA image (float64, range 0-1)
        background_color: Background color (RGB, range 0-255)
    
    Returns:
        RGB image (float64, range 0-1)
    """
    if len(img.shape) != 3 or img.shape[2] != 4:
        if len(img.shape) == 3 and img.shape[2] == 3:
            return img
        if len(img.shape) == 2:
            return np.stack([img] * 3, axis=-1)
        return img
    
    rgb = img[:, :, :3]
    alpha = img[:, :, 3:4]
    
    bg = np.array(background_color, dtype=np.float64) / 255.0
    bg = bg.reshape(1, 1, 3)
    
    composite = rgb * alpha + bg * (1 - alpha)
    
    return composite


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error (MSE)
    
    Args:
        img1, img2: Image arrays (float64, range 0-1)
    
    Returns:
        MSE value
    """
    mse = np.mean((img1 - img2) ** 2)
    return mse


def calculate_lpips(img1, img2, lpips_model):
    """Calculate LPIPS distance
    
    Args:
        img1, img2: Image arrays (float64, range 0-1, shape: HxWx3)
        lpips_model: LPIPS model instance
    
    Returns:
        LPIPS value
    """
    if not LPIPS_AVAILABLE:
        return None
    
    try:
        img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0)
        
        img1_tensor = img1_tensor * 2 - 1
        img2_tensor = img2_tensor * 2 - 1
        
        device = next(lpips_model.parameters()).device
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)
        
        with torch.no_grad():
            lpips_value = lpips_model(img1_tensor, img2_tensor)
        
        return lpips_value.item()
    except Exception as e:
        print(f"Error calculating LPIPS: {str(e)}")
        return None


def evaluate_images(gt_path, gen_path, use_white_background=True, 
                    background_color=(255, 255, 255), compute_lpips=True):
    """Calculate MSE, SSIM and LPIPS between two images
    
    Args:
        gt_path: Path to ground truth image
        gen_path: Path to generated image
        use_white_background: Whether to composite transparent images onto white background
        background_color: Background color (R, G, B)
        compute_lpips: Whether to compute LPIPS
        
    Returns:
        mse, ssim, lpips values (lpips may be None)
    """
    try:
        if not os.path.exists(gt_path):
            print(f"Error: File {gt_path} does not exist.")
            return None, None, None
        if not os.path.exists(gen_path):
            print(f"Error: File {gen_path} does not exist.")
            return None, None, None
        
        # Initialize LPIPS model
        lpips_model = None
        if compute_lpips and LPIPS_AVAILABLE:
            try:
                lpips_model = lpips.LPIPS(net='alex')
                if torch.cuda.is_available():
                    lpips_model = lpips_model.cuda()
                lpips_model.eval()
            except Exception as e:
                print(f"Warning: Could not load LPIPS model: {str(e)}")
                lpips_model = None
        
        # Read images
        img1 = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(gen_path, cv2.IMREAD_UNCHANGED)
        
        if img1 is None or img2 is None:
            print(f"Error: Could not read image {gt_path} or {gen_path}")
            return None, None, None
        
        # Ensure same dimensions
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to float64, range [0, 1]
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
        
        # Convert BGR to RGB
        if len(img1.shape) == 3:
            if img1.shape[2] == 3:
                img1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            elif img1.shape[2] == 4:
                img1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_BGRA2RGBA).astype(np.float64) / 255.0
        
        if len(img2.shape) == 3:
            if img2.shape[2] == 3:
                img2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            elif img2.shape[2] == 4:
                img2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_BGRA2RGBA).astype(np.float64) / 255.0
        
        # Process images based on settings
        if use_white_background:
            img1_cmp = composite_on_background(img1, background_color)
            img2_cmp = composite_on_background(img2, background_color)
        else:
            if len(img1.shape) == 3 and img1.shape[2] == 4:
                img1_cmp = img1[:, :, :3]
            else:
                img1_cmp = img1
            
            if len(img2.shape) == 3 and img2.shape[2] == 4:
                img2_cmp = img2[:, :, :3]
            else:
                img2_cmp = img2
        
        # Ensure RGB format
        if len(img1_cmp.shape) == 2:
            img1_cmp = np.stack([img1_cmp] * 3, axis=-1)
        if len(img2_cmp.shape) == 2:
            img2_cmp = np.stack([img2_cmp] * 3, axis=-1)
        
        # Calculate MSE
        mse_val = calculate_mse(img1_cmp, img2_cmp)
        
        # Calculate SSIM
        skimage_version = [int(x) for x in skimage.__version__.split('.')[:2]]
        use_multichannel = skimage_version[0] < 0 or (skimage_version[0] == 0 and skimage_version[1] < 19)
        
        if use_multichannel:
            ssim_val = ssim(img1_cmp, img2_cmp, multichannel=True, data_range=1.0)
        else:
            ssim_val = ssim(img1_cmp, img2_cmp, channel_axis=2, data_range=1.0)
        
        # Calculate LPIPS
        lpips_val = None
        if lpips_model is not None:
            lpips_val = calculate_lpips(img1_cmp, img2_cmp, lpips_model)
        
        return mse_val, ssim_val, lpips_val
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate MSE, SSIM and LPIPS between two images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  MSE    - Mean Squared Error (lower is better)
  SSIM   - Structural Similarity (higher is better, range 0-1)
  LPIPS  - Perceptual Similarity (lower is better, requires lpips library)

Examples:
  # Basic usage (default white background)
  python script.py --gt_img /path/to/gt.png --gen_img /path/to/gen.png
  
  # Without white background, compare RGB directly
  python script.py --gt_img /path/to/gt.png --gen_img /path/to/gen.png --no-white-bg
  
  # Custom background color (gray)
  python script.py --gt_img /path/to/gt.png --gen_img /path/to/gen.png --bg-color 128 128 128
  
  # Without LPIPS (saves time)
  python script.py --gt_img /path/to/gt.png --gen_img /path/to/gen.png --no-lpips

Dependencies:
  pip install scikit-image opencv-python numpy lpips torch torchvision
        """
    )
    
    parser.add_argument('--gt_img', required=True, help='Path to ground truth image')
    parser.add_argument('--gen_img', required=True, help='Path to generated image')
    parser.add_argument('--no-white-bg', action='store_true',
                        help='Do not use white background compositing, compare RGB channels directly')
    parser.add_argument('--bg-color', nargs=3, type=int, metavar=('R', 'G', 'B'),
                        default=[255, 255, 255],
                        help='Background color RGB values 0-255 (default: 255 255 255 white)')
    parser.add_argument('--no-lpips', action='store_true', dest='no_lpips',
                        help='Do not compute LPIPS (saves time and memory)')
    
    args = parser.parse_args()
    
    use_white_background = not args.no_white_bg
    compute_lpips = not args.no_lpips
    
    mse_val, ssim_val, lpips_val = evaluate_images(
        args.gt_img, 
        args.gen_img,
        use_white_background=use_white_background,
        background_color=tuple(args.bg_color),
        compute_lpips=compute_lpips
    )
    
    if mse_val is not None:
        print(f"MSE: {mse_val:.3f}")
        print(f"SSIM: {ssim_val:.3f}")
        if lpips_val is not None:
            print(f"LPIPS: {lpips_val:.3f}")
        else:
            print("LPIPS: Not computed")
            
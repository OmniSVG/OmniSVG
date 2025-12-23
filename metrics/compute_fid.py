import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from scipy import linalg
from tqdm import tqdm
import warnings
import tempfile
import shutil

# Ignore scipy warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


class InceptionV3Feature:
    """InceptionV3 feature extractor"""
    def __init__(self, device='cuda'):
        self.device = device
        # Load pretrained InceptionV3 model
        try:
            self.model = models.inception_v3(pretrained=True)
        except:
            # For newer versions of PyTorch
            self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        # Set to evaluation mode and remove classification layer
        self.model.eval()
        self.model.fc = torch.nn.Identity()
        self.model.to(device)
        
        # Image preprocessing
        self.preprocess = transforms.Compose([ 
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features_batch(self, image_paths, batch_size=50):
        """Extract image features in batches"""
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image = self.preprocess(image)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Stack into batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
            
            # Add to feature list
            all_features.append(batch_features.cpu())
            
            # Clear memory
            del batch_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not all_features:
            raise ValueError("Unable to extract features from images")
        
        # Merge all batch features
        all_features = torch.cat(all_features, dim=0).numpy()
        return all_features
    
    def extract_features_from_pil_images(self, images, batch_size=50):
        """Extract features directly from PIL images"""
        all_features = []
        
        for i in tqdm(range(0, len(images), batch_size)):
            batch_images_pil = images[i:i + batch_size]
            batch_images = []
            
            for image in batch_images_pil:
                try:
                    if not isinstance(image, Image.Image):
                        continue
                    image = image.convert('RGB')
                    image = self.preprocess(image)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Stack into batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
            
            # Add to feature list
            all_features.append(batch_features.cpu())
            
            # Clear memory
            del batch_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not all_features:
            raise ValueError("Unable to extract features from images")
        
        # Merge all batch features
        all_features = torch.cat(all_features, dim=0).numpy()
        return all_features


def download_gt_dataset(dataset_type='icon', sample_percentage=0.03, seed=42, cache_dir=None):
    """
    Download ground truth dataset from HuggingFace and sample 3% with fixed seed.
    
    Args:
        dataset_type: 'icon' for MMSVG-Icon or 'illustration' for MMSVG-Illustration
        sample_percentage: Percentage of data to sample (default 0.03 = 3%)
        seed: Random seed for reproducibility (default 42)
        cache_dir: Directory to cache the downloaded images
    
    Returns:
        List of PIL images
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Select dataset based on type
    dataset_map = {
        'icon': 'OmniSVG/MMSVG-Icon',
        'illustration': 'OmniSVG/MMSVG-Illustration'
    }
    
    if dataset_type.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from: {list(dataset_map.keys())}")
    
    dataset_name = dataset_map[dataset_type.lower()]
    print(f"Loading dataset: {dataset_name}...")
    
    # Load dataset
    ds = load_dataset(dataset_name)
    
    # Get the main split (usually 'train')
    if 'train' in ds:
        data = ds['train']
    else:
        # Use the first available split
        split_name = list(ds.keys())[0]
        data = ds[split_name]
        print(f"Using split: {split_name}")
    
    total_samples = len(data)
    sample_size = int(total_samples * sample_percentage)
    
    print(f"Total samples: {total_samples}")
    print(f"Sampling {sample_percentage*100:.1f}% = {sample_size} samples with seed={seed}")
    
    # Set random seed and sample
    np.random.seed(seed)
    sampled_indices = np.random.choice(total_samples, size=sample_size, replace=False)
    sampled_indices = sorted(sampled_indices)
    
    # Extract images
    print("Extracting images from dataset...")
    images = []
    
    # Find image column name
    image_columns = ['image', 'png', 'img', 'picture', 'rendered_image']
    image_col = None
    
    for col in image_columns:
        if col in data.column_names:
            image_col = col
            break
    
    if image_col is None:
        print(f"Available columns: {data.column_names}")
        raise ValueError("Could not find image column in dataset")
    
    print(f"Using image column: {image_col}")
    
    for idx in tqdm(sampled_indices, desc="Loading images"):
        try:
            img = data[int(idx)][image_col]
            if isinstance(img, Image.Image):
                images.append(img)
            elif isinstance(img, dict) and 'bytes' in img:
                # Handle bytes format
                import io
                img_bytes = img['bytes']
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
            elif isinstance(img, str):
                # Handle path format
                img = Image.open(img)
                images.append(img)
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            continue
    
    print(f"Successfully loaded {len(images)} images")
    return images


def save_images_to_folder(images, folder_path):
    """Save PIL images to a folder"""
    os.makedirs(folder_path, exist_ok=True)
    image_paths = []
    
    for i, img in enumerate(tqdm(images, desc="Saving images")):
        try:
            path = os.path.join(folder_path, f"gt_{i:06d}.png")
            img.convert('RGB').save(path)
            image_paths.append(path)
        except Exception as e:
            print(f"Error saving image {i}: {e}")
            continue
    
    return image_paths


def calculate_activation_statistics(features):
    """Calculate mean and covariance of features"""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet distance"""
    # Calculate squared distance between means
    diff = mu1 - mu2
    dot_product = np.sum(diff * diff)
    
    # Add eps to diagonal for numerical stability
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    
    # Calculate square root of covariance product
    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception as e:
        print(f"Error computing square root: {e}")
        print("Using eigenvalue decomposition method...")
        A = sigma1.dot(sigma2)
        eigenvalues, eigenvectors = linalg.eigh(A)
        eigenvalues = np.maximum(eigenvalues, 0)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        covmean = eigenvectors.dot(np.diag(sqrt_eigenvalues)).dot(eigenvectors.T)
    
    # Handle possible complex parts
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID formula
    return dot_product + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def find_image_files(folder):
    """Find all image files in folder"""
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    image_paths = []
    
    for root, _, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_paths.append(os.path.join(root, file))
    
    return image_paths


def calculate_fid(gt_folder, gen_folder, device='cuda', batch_size=50, sample_percentage=0.03):
    """Calculate FID between images in two folders"""
    # Check if folders exist
    if not os.path.exists(gt_folder):
        raise ValueError(f"Ground truth image folder '{gt_folder}' does not exist")
    if not os.path.exists(gen_folder):
        raise ValueError(f"Generated image folder '{gen_folder}' does not exist")
    
    # Initialize InceptionV3 feature extractor
    feature_extractor = InceptionV3Feature(device)
    
    # Get image paths
    print(f"Scanning images in {gt_folder}...")
    gt_image_paths = find_image_files(gt_folder)
    
    print(f"Scanning images in {gen_folder}...")
    gen_image_paths = find_image_files(gen_folder)
    
    print(f"Found {len(gt_image_paths)} ground truth images and {len(gen_image_paths)} generated images")
    
    if len(gt_image_paths) == 0:
        raise ValueError(f"No images found in ground truth folder '{gt_folder}'")
    if len(gen_image_paths) == 0:
        raise ValueError(f"No images found in generated folder '{gen_folder}'")
    
    # Sample percentage of ground truth images
    np.random.seed(42)  # Use fixed seed for reproducibility
    gt_image_paths = np.random.choice(gt_image_paths, size=int(len(gt_image_paths) * sample_percentage), replace=False)
    
    # Extract features
    print("Extracting features for ground truth images...")
    gt_features = feature_extractor.extract_features_batch(gt_image_paths, batch_size)
    
    print("Extracting features for generated images...")
    gen_features = feature_extractor.extract_features_batch(gen_image_paths, batch_size)
    
    # Calculate statistics
    print("Calculating statistics...")
    mu_gt, sigma_gt = calculate_activation_statistics(gt_features)
    mu_gen, sigma_gen = calculate_activation_statistics(gen_features)
    
    # Calculate FID value
    print("Calculating FID...")
    fid_value = calculate_frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
    
    return fid_value


def calculate_fid_with_hf_dataset(
    gen_folder, 
    dataset_type='icon', 
    device='cuda', 
    batch_size=50, 
    sample_percentage=0.03,
    seed=42,
    cache_dir=None
):
    """
    Calculate FID between generated images and HuggingFace dataset ground truth.
    
    Args:
        gen_folder: Path to generated images folder
        dataset_type: 'icon' for MMSVG-Icon or 'illustration' for MMSVG-Illustration
        device: Device to use (cuda or cpu)
        batch_size: Batch size for feature extraction
        sample_percentage: Percentage of GT data to sample (default 0.03 = 3%)
        seed: Random seed for reproducibility (default 42)
        cache_dir: Directory to cache downloaded images (optional)
    
    Returns:
        FID value
    """
    # Check if generated folder exists
    if not os.path.exists(gen_folder):
        raise ValueError(f"Generated image folder '{gen_folder}' does not exist")
    
    # Initialize InceptionV3 feature extractor
    feature_extractor = InceptionV3Feature(device)
    
    # Download and sample GT images from HuggingFace
    print(f"\n{'='*60}")
    print(f"Downloading Ground Truth Dataset: {dataset_type}")
    print(f"Sample percentage: {sample_percentage*100:.1f}%, Seed: {seed}")
    print(f"{'='*60}\n")
    
    gt_images = download_gt_dataset(
        dataset_type=dataset_type,
        sample_percentage=sample_percentage,
        seed=seed,
        cache_dir=cache_dir
    )
    
    if len(gt_images) == 0:
        raise ValueError("No ground truth images loaded from dataset")
    
    # Get generated image paths
    print(f"\nScanning images in {gen_folder}...")
    gen_image_paths = find_image_files(gen_folder)
    
    print(f"Found {len(gt_images)} ground truth images and {len(gen_image_paths)} generated images")
    
    if len(gen_image_paths) == 0:
        raise ValueError(f"No images found in generated folder '{gen_folder}'")
    
    # Extract features from GT images (PIL images)
    print("\nExtracting features for ground truth images...")
    gt_features = feature_extractor.extract_features_from_pil_images(gt_images, batch_size)
    
    # Clear GT images from memory
    del gt_images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Extract features from generated images
    print("\nExtracting features for generated images...")
    gen_features = feature_extractor.extract_features_batch(gen_image_paths, batch_size)
    
    # Calculate statistics
    print("\nCalculating statistics...")
    mu_gt, sigma_gt = calculate_activation_statistics(gt_features)
    mu_gen, sigma_gen = calculate_activation_statistics(gen_features)
    
    # Calculate FID value
    print("Calculating FID...")
    fid_value = calculate_frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
    
    return fid_value


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate FID between two image folders or with HuggingFace dataset')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'folder', 'hf'],
                        help='Mode: auto (auto-detect), folder (use local GT folder), hf (download from HuggingFace)')
    
    # Local folder mode
    parser.add_argument('--gt', type=str, default=None, help='Path to ground truth images folder (for folder mode)')
    
    # HuggingFace mode
    parser.add_argument('--dataset', type=str, default='icon', choices=['icon', 'illustration'],
                        help='Dataset type for HuggingFace mode: icon (MMSVG-Icon) or illustration (MMSVG-Illustration)')
    
    # Common arguments
    parser.add_argument('--gen', type=str, required=True, help='Path to generated images folder')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for feature extraction')
    parser.add_argument('--sample-percentage', type=float, default=0.03, 
                        help='Percentage of ground truth images to sample (default: 0.03 = 3%%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--cache-dir', type=str, default=None, help='Directory to cache downloaded images')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Determine mode
    if args.mode == 'auto':
        if args.gt and os.path.exists(args.gt):
            args.mode = 'folder'
            print("Auto-detected mode: folder (using local GT folder)")
        else:
            args.mode = 'hf'
            print("Auto-detected mode: hf (downloading from HuggingFace)")
    
    try:
        if args.mode == 'folder':
            if not args.gt:
                raise ValueError("--gt argument is required for folder mode")
            fid = calculate_fid(
                args.gt, 
                args.gen, 
                args.device, 
                args.batch_size, 
                args.sample_percentage
            )
        else:  # hf mode
            fid = calculate_fid_with_hf_dataset(
                gen_folder=args.gen,
                dataset_type=args.dataset,
                device=args.device,
                batch_size=args.batch_size,
                sample_percentage=args.sample_percentage,
                seed=args.seed,
                cache_dir=args.cache_dir
            )
        
        print(f"\n{'='*60}")
        print(f"FID: {fid:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error calculating FID: {e}")
        import traceback
        traceback.print_exc()

# Icon 
#python compute_fid.py --gen /path/to/generated_images --dataset icon

# Illustration 
#python compute_fid.py --gen /path/to/generated_images --dataset illustration

# python compute_fid.py --mode folder --gt /path/to/gt_images --gen /path/to/generated_images

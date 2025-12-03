import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from scipy import linalg
from tqdm import tqdm
import warnings

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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate FID between two image folders')
    parser.add_argument('--gt', type=str, default='/path/to/gt_images', help='Path to ground truth images folder')
    parser.add_argument('--gen', type=str, default='/path/to/gen_images', help='Path to generated images folder')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for feature extraction')
    parser.add_argument('--sample-percentage', type=float, default=0.03, help='Percentage of ground truth images to sample')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    try:
        fid = calculate_fid(args.gt, args.gen, args.device, args.batch_size, args.sample_percentage)
        print(f"FID: {fid:.4f}")
    except Exception as e:
        print(f"Error calculating FID: {e}")
        import traceback
        traceback.print_exc()
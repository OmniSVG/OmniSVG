import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import clip
from PIL import Image, ImageFile
import argparse
import urllib.request

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Aesthetic predictor checkpoint configuration
AESTHETIC_MODEL_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "aesthetic_predictor")
AESTHETIC_MODEL_PATH = os.path.join(AESTHETIC_MODEL_CACHE, "sac+logos+ava1-l14-linearMSE.pth")

# MLP model architecture
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def download_aesthetic_model():
    """Download aesthetic predictor model if not exists."""
    if not os.path.exists(AESTHETIC_MODEL_PATH):
        os.makedirs(AESTHETIC_MODEL_CACHE, exist_ok=True)
        print(f"Downloading aesthetic predictor model...")
        print(f"URL: {AESTHETIC_MODEL_URL}")
        urllib.request.urlretrieve(AESTHETIC_MODEL_URL, AESTHETIC_MODEL_PATH)
        print(f"Model saved to: {AESTHETIC_MODEL_PATH}")
    return AESTHETIC_MODEL_PATH

def compute_aesthetic_score(image_path, prompt, device):
    """Compute the aesthetic score for a single image."""
    # Download (if needed) and load aesthetic predictor model
    model_path = download_aesthetic_model()
    model = MLP(768)  # CLIP embedding dim is 768 for ViT-L/14
    s = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(s)
    model.to(device)
    model.eval()
    
    # Load CLIP model (automatically downloads if not cached)
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    
    # Load and process image
    pil_image = Image.open(image_path).convert("RGB")
    image = preprocess(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get image features
        image_features = clip_model.encode_image(image)
        
        # Compute aesthetic score
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        tensor_type = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
        aesthetic_score = model(torch.from_numpy(im_emb_arr).to(device).type(tensor_type))
        
        # Compute CLIP similarity score (text-image alignment)
        text_tokens = clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        clip_score = (image_features @ text_features.T).squeeze().cpu().item()
    
    return aesthetic_score.cpu().item(), clip_score

def main():
    parser = argparse.ArgumentParser(description="Compute aesthetic score for a single image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for CLIP similarity")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        aesthetic_score, clip_score = compute_aesthetic_score(args.image, args.prompt, device)
        print(f"Aesthetic Score: {aesthetic_score:.4f}")
        print(f"CLIP Score: {clip_score:.4f}")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
